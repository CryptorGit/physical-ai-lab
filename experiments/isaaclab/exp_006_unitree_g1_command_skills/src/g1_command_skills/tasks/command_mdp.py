"""Stateful, robot-local motion commands for staged locomotion skills."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class SkillId(IntEnum):
    RUN = 0
    STOP = 1
    TURN = 2
    CROUCH = 3
    STEP_OVER = 4
    LAND = 5


class EpisodeKind(IntEnum):
    RUN = 0
    TURN = 1
    STOP = 2
    SEQUENCE = 3
    CROUCH = 4


SKILL_COUNT = len(SkillId)
EXTRA_COMMAND_DIM = 29
LEGACY_OBSERVATION_DIM = 123
POLICY_OBSERVATION_DIM = LEGACY_OBSERVATION_DIM + EXTRA_COMMAND_DIM


def _ensure_mutable_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a normal tensor suitable for in-place command-state updates."""
    if torch.is_inference(tensor):
        with torch.inference_mode(False):
            return tensor.clone()
    return tensor


class MotionCommand(UniformVelocityCommand):
    """Generate one skill at a time and expose continuous, local parameters.

    ``command`` deliberately remains the legacy 3-D body-velocity tensor.  The
    additional 29 values are appended by a separate observation term, preserving
    every exp_005 observation column and enabling an exact structural transplant.
    """

    cfg: object

    _MUTABLE_STATE_NAMES = (
        "skill_id", "previous_skill_id", "segment_index", "episode_kind",
        "segment_elapsed", "segment_duration", "transition_progress", "target_speed",
        "target_heading_w", "target_position_w", "target_pelvis_height", "obstacle_geometry_b",
        "target_vertical_velocity", "recovery_mode", "target_posture_rp", "path_origin_w",
        "path_heading_w", "path_lateral_error", "path_forward_velocity", "path_lateral_velocity",
        "path_curvature", "path_lookahead_b", "turn_start_heading_w", "commanded_turn_angle_rad",
        "actual_accumulated_yaw_rad", "_turn_previous_heading_w", "heading_error",
        "target_displacement_b", "stop_entry_speed", "stop_initial_distance",
        "stop_target_heading_w",
        "stop_required_deceleration", "stop_braking_target_speed", "stop_progress",
        "stop_hold_elapsed", "stop_hold_progress", "stop_hold_complete", "extra_command",
        "crouch_entry_height", "crouch_commanded_drop", "crouch_down_duration",
        "crouch_requested_depth_m", "crouch_applied_depth_m", "crouch_command_supported",
        "crouch_command_clamped", "crouch_unsupported_reason_code",
        "crouch_hold_duration", "crouch_return_duration", "crouch_stand_hold_duration",
        "crouch_phase", "crouch_phase_progress", "crouch_hold_progress",
        "crouch_return_progress", "crouch_hold_complete", "crouch_return_complete",
        "crouch_stand_hold_complete", "crouch_current_height_error", "joint_limit_proximity",
        "crouch_settle_streak", "crouch_base_ready_streak", "crouch_settle_success",
        "crouch_settle_failure", "crouch_entry_height_fixed", "crouch_settle_time",
        "crouch_motion_start_time", "crouch_settle_height_min", "crouch_settle_height_max",
        "crouch_down_entry_speed", "crouch_base_transition_started", "crouch_base_transition_progress",
        "_just_reset", "vel_command_b", "heading_target", "is_standing_env", "is_heading_env",
    )

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n = self.num_envs
        # Command state is mutated throughout reset/resample/update. Explicitly
        # opt out even if a caller accidentally constructs the environment in
        # an outer inference-mode context.
        with torch.inference_mode(False):
            self.skill_id = torch.zeros(n, dtype=torch.long, device=self.device)
            self.previous_skill_id = torch.zeros_like(self.skill_id)
            self.segment_index = torch.zeros_like(self.skill_id)
            self.episode_kind = torch.zeros_like(self.skill_id)
            self.segment_elapsed = torch.zeros(n, device=self.device)
            self.segment_duration = torch.ones(n, device=self.device)
            self.transition_progress = torch.ones(n, device=self.device)
            self.target_speed = torch.zeros(n, device=self.device)
            self.target_heading_w = torch.zeros(n, device=self.device)
            self.target_position_w = torch.zeros((n, 2), device=self.device)
            self.target_pelvis_height = torch.full((n,), float(cfg.standing_pelvis_height_m), device=self.device)
            self.obstacle_geometry_b = torch.zeros((n, 4), device=self.device)
            self.target_vertical_velocity = torch.zeros(n, device=self.device)
            self.recovery_mode = torch.zeros(n, device=self.device)
            self.target_posture_rp = torch.zeros((n, 2), device=self.device)
            self.path_origin_w = torch.zeros((n, 2), device=self.device)
            self.path_heading_w = torch.zeros(n, device=self.device)
            self.path_lateral_error = torch.zeros(n, device=self.device)
            self.path_forward_velocity = torch.zeros(n, device=self.device)
            self.path_lateral_velocity = torch.zeros(n, device=self.device)
            self.path_curvature = torch.zeros(n, device=self.device)
            self.path_lookahead_b = torch.zeros((n, 2), device=self.device)
            self.turn_start_heading_w = torch.zeros(n, device=self.device)
            self.commanded_turn_angle_rad = torch.zeros(n, device=self.device)
            self.actual_accumulated_yaw_rad = torch.zeros(n, device=self.device)
            self._turn_previous_heading_w = torch.zeros(n, device=self.device)
            self.heading_error = torch.zeros(n, device=self.device)
            self.target_displacement_b = torch.zeros((n, 2), device=self.device)
            self.stop_entry_speed = torch.zeros(n, device=self.device)
            self.stop_target_heading_w = torch.zeros(n, device=self.device)
            self.stop_initial_distance = torch.zeros(n, device=self.device)
            self.stop_required_deceleration = torch.zeros(n, device=self.device)
            self.stop_braking_target_speed = torch.zeros(n, device=self.device)
            self.stop_progress = torch.zeros(n, device=self.device)
            self.stop_hold_elapsed = torch.zeros(n, device=self.device)
            self.stop_hold_progress = torch.zeros(n, device=self.device)
            self.stop_hold_complete = torch.zeros(n, dtype=torch.bool, device=self.device)
            self.crouch_entry_height = torch.zeros(n, device=self.device)
            self.crouch_commanded_drop = torch.zeros(n, device=self.device)
            self.crouch_requested_depth_m = torch.zeros(n, device=self.device)
            self.crouch_applied_depth_m = torch.zeros(n, device=self.device)
            self.crouch_command_supported = torch.ones(n, dtype=torch.bool, device=self.device)
            self.crouch_command_clamped = torch.zeros(n, dtype=torch.bool, device=self.device)
            self.crouch_unsupported_reason_code = torch.zeros(n, dtype=torch.long, device=self.device)
            self.crouch_down_duration = torch.ones(n, device=self.device)
            self.crouch_hold_duration = torch.ones(n, device=self.device)
            self.crouch_return_duration = torch.ones(n, device=self.device)
            self.crouch_stand_hold_duration = torch.ones(n, device=self.device)
            self.crouch_phase = torch.zeros(n, dtype=torch.long, device=self.device)
            self.crouch_phase_progress = torch.zeros(n, device=self.device)
            self.crouch_hold_progress = torch.zeros(n, device=self.device)
            self.crouch_return_progress = torch.zeros(n, device=self.device)
            self.crouch_hold_complete = torch.zeros(n, dtype=torch.bool, device=self.device)
            self.crouch_return_complete = torch.zeros(n, dtype=torch.bool, device=self.device)
            self.crouch_stand_hold_complete = torch.zeros(n, dtype=torch.bool, device=self.device)
            self.crouch_current_height_error = torch.zeros(n, device=self.device)
            self.joint_limit_proximity = torch.zeros(n, device=self.device)
            self.crouch_settle_streak = torch.zeros(n, dtype=torch.long, device=self.device)
            self.crouch_base_ready_streak = torch.zeros(n, dtype=torch.long, device=self.device)
            self.crouch_settle_success = torch.zeros(n, dtype=torch.bool, device=self.device)
            self.crouch_settle_failure = torch.zeros(n, dtype=torch.bool, device=self.device)
            self.crouch_entry_height_fixed = torch.zeros(n, dtype=torch.bool, device=self.device)
            self.crouch_settle_time = torch.zeros(n, device=self.device)
            self.crouch_motion_start_time = torch.zeros(n, device=self.device)
            self.crouch_settle_height_min = torch.zeros(n, device=self.device)
            self.crouch_settle_height_max = torch.zeros(n, device=self.device)
            self.crouch_down_entry_speed = torch.zeros(n, device=self.device)
            self.crouch_base_transition_started = torch.zeros(n, dtype=torch.bool, device=self.device)
            self.crouch_base_transition_progress = torch.zeros(n, device=self.device)
            self.extra_command = torch.zeros((n, EXTRA_COMMAND_DIM), device=self.device)
            self._just_reset = torch.ones(n, dtype=torch.bool, device=self.device)
        self._turn_sample_count = 0
        self._crouch_sample_count = 0
        self._crouch_joint_ids, _ = self.robot.find_joints(".*_(hip_pitch|knee|ankle_pitch)_joint")
        contact = self._env.scene.sensors["contact_forces"]
        _, foot_names = self.robot.find_bodies(".*_ankle_roll_link")
        self._crouch_contact_sensor = contact
        self._crouch_sensor_foot_ids = [contact.body_names.index(name) for name in foot_names]
        self._ensure_mutable_state()

    def _ensure_mutable_state(self) -> None:
        """One-time/reset-time insurance for old or externally-created state."""
        for name in self._MUTABLE_STATE_NAMES:
            tensor = getattr(self, name, None)
            if isinstance(tensor, torch.Tensor):
                setattr(self, name, _ensure_mutable_tensor(tensor))

    def _ids(self, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

    def _sample(self, count: int, bounds: tuple[float, float]) -> torch.Tensor:
        return torch.empty(count, device=self.device).uniform_(*bounds)

    def _choose_turn_delta(self, count: int) -> torch.Tensor:
        angles = torch.tensor(self.cfg.turn_angles_deg, dtype=torch.float, device=self.device)
        if bool(getattr(self.cfg, "deterministic_turn_evaluation", False)):
            signed_angles = torch.stack((angles, -angles), dim=1).flatten()
            indices = (torch.arange(count, device=self.device) + self._turn_sample_count) % len(signed_angles)
            self._turn_sample_count += count
            return torch.deg2rad(signed_angles[indices])
        probabilities = torch.tensor(self.cfg.turn_angle_probabilities, dtype=torch.float, device=self.device)
        if probabilities.shape != angles.shape or not torch.isclose(
            probabilities.sum(), torch.tensor(1.0, device=self.device)
        ):
            raise ValueError(
                f"turn_angle_probabilities must match turn_angles_deg and sum to 1: "
                f"{self.cfg.turn_angle_probabilities}"
            )
        choices = angles[torch.multinomial(probabilities, count, replacement=True)]
        direction_probabilities = torch.tensor(
            self.cfg.turn_direction_probabilities, dtype=torch.float, device=self.device
        )
        if direction_probabilities.shape != (2,) or not torch.isclose(
            direction_probabilities.sum(), torch.tensor(1.0, device=self.device)
        ):
            raise ValueError(
                "turn_direction_probabilities must be (left, right) and sum to 1: "
                f"{self.cfg.turn_direction_probabilities}"
            )
        directions = torch.multinomial(direction_probabilities, count, replacement=True)
        signs = torch.where(directions == 0, 1.0, -1.0)
        return torch.deg2rad(choices) * signs

    def _choose_crouch_depth(self, count: int) -> torch.Tensor:
        depths = tuple(float(value) for value in getattr(self.cfg, "crouch_evaluation_depths_m", ()))
        if not depths:
            return self._sample(count, self.cfg.crouch_height_drop_range_m)
        values = torch.tensor(depths, dtype=torch.float, device=self.device)
        indices = (torch.arange(count, device=self.device) + self._crouch_sample_count) % len(values)
        self._crouch_sample_count += count
        return values[indices]

    def _set_skill(self, ids: torch.Tensor, skill: SkillId, duration: float, *, initial: bool = False) -> None:
        if not initial:
            self.previous_skill_id[ids] = self.skill_id[ids]
        else:
            self.previous_skill_id[ids] = int(skill)
        self.skill_id[ids] = int(skill)
        self.segment_elapsed[ids] = 0.0
        jitter = float(self.cfg.phase_duration_jitter_fraction)
        if jitter > 0.0:
            self.segment_duration[ids] = duration * self._sample(len(ids), (1.0 - jitter, 1.0 + jitter))
        else:
            self.segment_duration[ids] = duration
        self.transition_progress[ids] = 1.0 if initial else 0.0

    def _configure_skill_targets(self, ids: torch.Tensor, skill: SkillId) -> None:
        count = len(ids)
        heading = self.robot.data.heading_w.torch[ids]
        position = self.robot.data.root_pos_w.torch[ids, :2]
        if skill == SkillId.RUN:
            self.target_position_w[ids] = position
            self.target_speed[ids] = self._sample(count, self.cfg.run_speed_range)
            if (self.segment_index[ids] == 0).any():
                self.target_heading_w[ids] = heading
            self.path_heading_w[ids] = self.target_heading_w[ids]
            initial_lateral_error = self._sample(count, self.cfg.run_path_initial_lateral_error_range_m)
            normal_x = -torch.sin(self.path_heading_w[ids])
            normal_y = torch.cos(self.path_heading_w[ids])
            self.path_origin_w[ids, 0] = position[:, 0] - normal_x * initial_lateral_error
            self.path_origin_w[ids, 1] = position[:, 1] - normal_y * initial_lateral_error
            self.path_curvature[ids] = 0.0
        elif skill == SkillId.TURN:
            self.obstacle_geometry_b[ids].zero_()
            self.target_position_w[ids] = position
            self.target_speed[ids] = self._sample(count, self.cfg.turn_speed_range)
            turn_delta = self._choose_turn_delta(count)
            self.turn_start_heading_w[ids] = heading
            self.commanded_turn_angle_rad[ids] = turn_delta
            self.actual_accumulated_yaw_rad[ids] = 0.0
            self._turn_previous_heading_w[ids] = heading
            self.target_heading_w[ids] = wrap_to_pi(heading + turn_delta)
        elif skill == SkillId.STOP:
            self.obstacle_geometry_b[ids].zero_()
            distance = self._sample(count, self.cfg.stop_distance_range)
            entry_speed = self.robot.data.root_lin_vel_b.torch[ids, 0].clamp_min(0.0)
            entry_speed = torch.maximum(entry_speed, torch.full_like(entry_speed, float(self.cfg.stop_entry_speed_floor_mps)))
            effective_distance = (distance - float(self.cfg.stop_target_radius_m)).clamp_min(1.0e-3)
            deceleration = entry_speed.square() / (2.0 * effective_distance)
            deceleration.clamp_(float(self.cfg.stop_deceleration_range_mps2[0]), float(self.cfg.stop_deceleration_range_mps2[1]))
            self.stop_entry_speed[ids] = entry_speed
            self.stop_target_heading_w[ids] = heading
            self.stop_initial_distance[ids] = distance
            self.stop_required_deceleration[ids] = deceleration
            self.stop_braking_target_speed[ids] = entry_speed
            self.stop_progress[ids] = 0.0
            self.stop_hold_elapsed[ids] = 0.0
            self.stop_hold_progress[ids] = 0.0
            self.stop_hold_complete[ids] = False
            self.target_speed[ids] = entry_speed
            self.target_position_w[ids, 0] = position[:, 0] + distance * torch.cos(heading)
            self.target_position_w[ids, 1] = position[:, 1] + distance * torch.sin(heading)
            self.target_heading_w[ids] = heading
        elif skill == SkillId.CROUCH:
            entry_height = self.robot.data.root_pos_w.torch[ids, 2]
            self.target_position_w[ids] = position
            self.target_heading_w[ids] = heading
            self.target_speed[ids] = 0.0
            self.crouch_entry_height[ids] = entry_height
            requested_depth = self._choose_crouch_depth(count)
            self._set_crouch_command_contract(ids, requested_depth)
            self.crouch_down_duration[ids] = self._sample(count, self.cfg.crouch_down_time_range_s)
            self.crouch_hold_duration[ids] = self._sample(count, self.cfg.crouch_hold_time_range_s)
            self.crouch_return_duration[ids] = self._sample(count, self.cfg.crouch_return_time_range_s)
            self.crouch_stand_hold_duration[ids] = self._sample(count, self.cfg.crouch_stand_hold_time_range_s)
            self.segment_duration[ids] = (
                float(self.cfg.crouch_settle_timeout_s) + self.crouch_down_duration[ids] + self.crouch_hold_duration[ids]
                + self.crouch_return_duration[ids] + self.crouch_stand_hold_duration[ids]
            )
            self.crouch_phase[ids] = 0
            self.crouch_phase_progress[ids] = 0.0
            self.crouch_hold_progress[ids] = 0.0
            self.crouch_return_progress[ids] = 0.0
            self.crouch_hold_complete[ids] = False
            self.crouch_return_complete[ids] = False
            self.crouch_stand_hold_complete[ids] = False
            self.crouch_settle_streak[ids] = 0
            self.crouch_base_ready_streak[ids] = 0
            self.crouch_settle_success[ids] = False
            self.crouch_settle_failure[ids] = False
            self.crouch_entry_height_fixed[ids] = False
            self.crouch_settle_time[ids] = 0.0
            self.crouch_motion_start_time[ids] = 0.0
            self.crouch_settle_height_min[ids] = entry_height
            self.crouch_settle_height_max[ids] = entry_height
            self.crouch_down_entry_speed[ids] = 0.0
            standalone = bool(self.cfg.crouch_standalone_standing_base)
            self.crouch_base_transition_started[ids] = standalone
            self.crouch_base_transition_progress[ids] = 1.0 if standalone else 0.0
            self.target_pelvis_height[ids] = entry_height
            self.target_vertical_velocity[ids] = 0.0

    def _set_crouch_command_contract(self, ids: torch.Tensor, requested_depth: torch.Tensor) -> None:
        """Validate CROUCH depth without silently turning deep commands into shallow successes."""
        minimum = float(getattr(self.cfg, "crouch_supported_depth_min_m", 0.08))
        maximum = float(getattr(self.cfg, "crouch_supported_depth_max_m", 0.10))
        mode = str(getattr(self.cfg, "crouch_unsupported_command_mode", "reject"))
        supported = (requested_depth >= minimum) & (requested_depth <= maximum)
        if mode not in {"reject", "debug_clamp"}:
            raise ValueError(f"Unknown CROUCH unsupported-command mode: {mode}")
        if mode == "debug_clamp":
            applied = requested_depth.clamp(minimum, maximum)
            clamped = ~supported
        else:
            applied = torch.where(supported, requested_depth, torch.zeros_like(requested_depth))
            clamped = torch.zeros_like(supported)
        reason = torch.zeros_like(ids)
        reason[requested_depth < minimum] = 1
        reason[requested_depth > maximum] = 2
        self.crouch_requested_depth_m[ids] = requested_depth
        self.crouch_applied_depth_m[ids] = applied
        self.crouch_commanded_drop[ids] = applied
        self.crouch_command_supported[ids] = supported
        self.crouch_command_clamped[ids] = clamped
        self.crouch_unsupported_reason_code[ids] = reason

    def _resample_command(self, env_ids: Sequence[int]):
        with torch.inference_mode(False):
            self._ensure_mutable_state()
            return self._resample_command_mutable(env_ids)

    def _resample_command_mutable(self, env_ids: Sequence[int]):
        ids = self._ids(env_ids)
        if len(ids) == 0:
            return
        self.segment_index[ids] = 0
        probabilities = torch.tensor(self.cfg.rehearsal_probabilities, device=self.device, dtype=torch.float)
        if probabilities.shape != (5,) or not torch.isclose(probabilities.sum(), torch.tensor(1.0, device=self.device)):
            raise ValueError(f"Invalid rehearsal probabilities: {self.cfg.rehearsal_probabilities}")
        self.episode_kind[ids] = torch.multinomial(probabilities, len(ids), replacement=True)
        for kind in EpisodeKind:
            selected = ids[self.episode_kind[ids] == int(kind)]
            if len(selected) == 0:
                continue
            skills, durations = self._script(kind)
            self._set_skill(selected, skills[0], durations[0], initial=True)
            self._configure_skill_targets(selected, skills[0])
        self._just_reset[ids] = True
        self.is_standing_env[ids] = (self.skill_id[ids] == int(SkillId.STOP)) | (self.skill_id[ids] == int(SkillId.CROUCH))
        self.is_heading_env[ids] = True
        self._update_local_targets()
        self._update_extra_command()

    def _script(self, kind: EpisodeKind) -> tuple[tuple[SkillId, ...], tuple[float, ...]]:
        if kind == EpisodeKind.RUN:
            return (SkillId.RUN,), (float(self.cfg.single_skill_duration_s),)
        if kind == EpisodeKind.TURN:
            return (SkillId.RUN, SkillId.TURN, SkillId.RUN), tuple(float(v) for v in self.cfg.turn_script_durations_s)
        if kind == EpisodeKind.STOP:
            return (SkillId.RUN, SkillId.STOP), tuple(float(v) for v in self.cfg.stop_script_durations_s)
        if kind == EpisodeKind.CROUCH:
            return (SkillId.CROUCH,), (float(self.cfg.crouch_nominal_duration_s),)
        return (
            (SkillId.RUN, SkillId.TURN, SkillId.RUN, SkillId.STOP),
            tuple(float(v) for v in self.cfg.sequence_durations_s),
        )

    def _advance_scripts(self) -> None:
        for kind in EpisodeKind:
            kind_ids = (self.episode_kind == int(kind)).nonzero(as_tuple=False).flatten()
            if len(kind_ids) == 0:
                continue
            skills, durations = self._script(kind)
            expired = self.segment_elapsed[kind_ids] >= self.segment_duration[kind_ids]
            can_advance = self.segment_index[kind_ids] < len(skills) - 1
            ids = kind_ids[expired & can_advance]
            if len(ids) == 0:
                continue
            self.segment_index[ids] += 1
            for segment in range(1, len(skills)):
                selected = ids[self.segment_index[ids] == segment]
                if len(selected) > 0:
                    self._set_skill(selected, skills[segment], durations[segment])
                    self._configure_skill_targets(selected, skills[segment])

    def _update_local_targets(self) -> None:
        heading = self.robot.data.heading_w.torch
        self.heading_error = wrap_to_pi(self.target_heading_w - heading)
        delta_w = self.target_position_w - self.robot.data.root_pos_w.torch[:, :2]
        cos_h, sin_h = torch.cos(heading), torch.sin(heading)
        self.target_displacement_b[:, 0] = cos_h * delta_w[:, 0] + sin_h * delta_w[:, 1]
        self.target_displacement_b[:, 1] = -sin_h * delta_w[:, 0] + cos_h * delta_w[:, 1]
        self._update_path_targets(cos_h, sin_h)
        turn = self.skill_id == int(SkillId.TURN)
        self.target_displacement_b[turn, 0] = self.commanded_turn_angle_rad[turn]
        self.target_displacement_b[turn, 1] = self.actual_accumulated_yaw_rad[turn]
        stop = self.skill_id == int(SkillId.STOP)
        self.heading_error[stop] = wrap_to_pi(self.stop_target_heading_w[stop] - heading[stop])

    def _update_stop_targets(self, dt: float) -> None:
        """Update the closed-loop braking curve while keeping the entry goal fixed."""
        stop = self.skill_id == int(SkillId.STOP)
        if not stop.any():
            return
        remaining = self.target_displacement_b[:, 0]
        effective_remaining = torch.relu(remaining - float(self.cfg.stop_target_radius_m))
        braking = torch.sqrt(2.0 * self.stop_required_deceleration * effective_remaining)
        braking = torch.minimum(braking, self.stop_entry_speed)
        braking = torch.where(remaining > 0.0, braking, torch.zeros_like(braking))
        self.stop_braking_target_speed[stop] = braking[stop]
        self.target_speed[stop] = braking[stop]
        initial = self.stop_initial_distance.clamp_min(1.0e-6)
        self.stop_progress[stop] = (1.0 - torch.relu(remaining[stop]) / initial[stop]).clamp(0.0, 1.0)

        speed = torch.linalg.norm(self.robot.data.root_lin_vel_b.torch[:, :2], dim=1)
        holding = stop & (remaining.abs() <= float(self.cfg.stop_hold_position_tolerance_m)) & (
            speed <= float(self.cfg.stop_hold_speed_tolerance_mps)
        )
        self.stop_hold_elapsed[holding] += dt
        self.stop_hold_elapsed[stop & ~holding] = 0.0
        hold_s = max(float(self.cfg.stop_hold_duration_s), 1.0e-6)
        self.stop_hold_progress[stop] = (self.stop_hold_elapsed[stop] / hold_s).clamp(0.0, 1.0)
        self.stop_hold_complete[stop] = self.stop_hold_progress[stop] >= 1.0

    def _update_turn_progress(self) -> None:
        """Accumulate unwrapped yaw from the fixed heading at TURN entry."""
        heading = self.robot.data.heading_w.torch
        turn = self.skill_id == int(SkillId.TURN)
        delta = wrap_to_pi(heading - self._turn_previous_heading_w)
        self.actual_accumulated_yaw_rad[turn] += delta[turn]
        self._turn_previous_heading_w.copy_(heading)

    @staticmethod
    def _minimum_jerk(progress: torch.Tensor) -> torch.Tensor:
        progress = progress.clamp(0.0, 1.0)
        return progress**3 * (10.0 - 15.0 * progress + 6.0 * progress.square())

    def _update_crouch_targets(self) -> None:
        """Safety-settle, then track a relative down/hold/return trajectory."""
        crouch = self.skill_id == int(SkillId.CROUCH)
        if not crouch.any():
            return
        elapsed = self.segment_elapsed
        height = self.robot.data.root_pos_w.torch[:, 2]
        horizontal_speed = self.robot.data.root_lin_vel_b.torch[:, :2].norm(dim=1)
        vertical_speed = self.robot.data.root_lin_vel_w.torch[:, 2].abs()
        gravity = self.robot.data.projected_gravity_b.torch
        roll = torch.atan2(gravity[:, 1], -gravity[:, 2]).abs()
        pitch = torch.atan2(
            -gravity[:, 0], torch.sqrt(gravity[:, 1].square() + gravity[:, 2].square())
        ).abs()
        forces = self._crouch_contact_sensor.data.net_forces_w_history.torch[
            :, :, self._crouch_sensor_foot_ids, :
        ]
        double_support = (forces.norm(dim=-1).amax(dim=1) > 5.0).all(dim=1)
        safe = (
            (horizontal_speed <= float(self.cfg.crouch_settle_horizontal_speed_mps))
            & (vertical_speed <= float(self.cfg.crouch_settle_vertical_speed_mps))
            & (roll <= float(self.cfg.crouch_settle_tilt_rad))
            & (pitch <= float(self.cfg.crouch_settle_tilt_rad))
            & double_support
        )
        pending = crouch & ~self.crouch_settle_success & ~self.crouch_settle_failure
        self.crouch_settle_height_min[pending] = torch.minimum(self.crouch_settle_height_min[pending], height[pending])
        self.crouch_settle_height_max[pending] = torch.maximum(self.crouch_settle_height_max[pending], height[pending])

        # Integrated sequences defer running->standing cross-fade until the
        # same safety envelope is held. Isolated Stage A begins at progress 1.
        waiting_base = pending & ~self.crouch_base_transition_started
        self.crouch_base_ready_streak[waiting_base] = torch.where(
            safe[waiting_base], self.crouch_base_ready_streak[waiting_base] + 1,
            torch.zeros_like(self.crouch_base_ready_streak[waiting_base]),
        )
        required_steps = max(1, round(float(self.cfg.crouch_settle_hold_s) / float(self._env.step_dt)))
        start_transition = waiting_base & (self.crouch_base_ready_streak >= required_steps)
        self.crouch_base_transition_started[start_transition] = True
        transitioning = pending & self.crouch_base_transition_started & (self.crouch_base_transition_progress < 1.0)
        self.crouch_base_transition_progress[transitioning] = (
            self.crouch_base_transition_progress[transitioning]
            + float(self._env.step_dt) / max(float(self.cfg.crouch_base_crossfade_duration_s), 1.0e-6)
        ).clamp(max=1.0)

        waiting_settle = pending & (self.crouch_base_transition_progress >= 1.0)
        self.crouch_settle_streak[waiting_settle] = torch.where(
            safe[waiting_settle], self.crouch_settle_streak[waiting_settle] + 1,
            torch.zeros_like(self.crouch_settle_streak[waiting_settle]),
        )
        newly_settled = waiting_settle & (self.crouch_settle_streak >= required_steps)
        self.crouch_settle_success[newly_settled] = True
        self.crouch_entry_height_fixed[newly_settled] = True
        self.crouch_entry_height[newly_settled] = height[newly_settled]
        self.crouch_settle_time[newly_settled] = elapsed[newly_settled]
        self.crouch_motion_start_time[newly_settled] = elapsed[newly_settled]
        self.crouch_down_entry_speed[newly_settled] = horizontal_speed[newly_settled]
        motion_duration = (
            self.crouch_down_duration + self.crouch_hold_duration
            + self.crouch_return_duration + self.crouch_stand_hold_duration
        )
        self.segment_duration[newly_settled] = elapsed[newly_settled] + motion_duration[newly_settled]
        timed_out = pending & ~newly_settled & (elapsed >= float(self.cfg.crouch_settle_timeout_s))
        self.crouch_settle_failure[timed_out] = True

        not_moving = crouch & ~self.crouch_settle_success
        self.crouch_phase[not_moving] = 0
        self.crouch_phase_progress[not_moving] = 0.0
        self.target_pelvis_height[not_moving] = height[not_moving]
        self.target_vertical_velocity[not_moving] = 0.0
        motion_elapsed = (elapsed - self.crouch_motion_start_time).clamp_min(0.0)
        down_end = self.crouch_down_duration
        hold_end = down_end + self.crouch_hold_duration
        return_end = hold_end + self.crouch_return_duration
        total = return_end + self.crouch_stand_hold_duration
        moving = crouch & self.crouch_settle_success & self.crouch_command_supported
        down = moving & (motion_elapsed < down_end)
        hold = moving & (motion_elapsed >= down_end) & (motion_elapsed < hold_end)
        returning = moving & (motion_elapsed >= hold_end) & (motion_elapsed < return_end)
        stand_hold = moving & (motion_elapsed >= return_end)

        down_p = (motion_elapsed / self.crouch_down_duration.clamp_min(1.0e-6)).clamp(0.0, 1.0)
        hold_p = ((motion_elapsed - down_end) / self.crouch_hold_duration.clamp_min(1.0e-6)).clamp(0.0, 1.0)
        return_p = ((motion_elapsed - hold_end) / self.crouch_return_duration.clamp_min(1.0e-6)).clamp(0.0, 1.0)
        stand_p = ((motion_elapsed - return_end) / self.crouch_stand_hold_duration.clamp_min(1.0e-6)).clamp(0.0, 1.0)
        down_curve = self._minimum_jerk(down_p)
        return_curve = self._minimum_jerk(return_p)
        relative_target = torch.zeros_like(elapsed)
        relative_target[down] = -self.crouch_commanded_drop[down] * down_curve[down]
        relative_target[hold] = -self.crouch_commanded_drop[hold]
        relative_target[returning] = -self.crouch_commanded_drop[returning] * (1.0 - return_curve[returning])
        self.target_pelvis_height[crouch] = self.crouch_entry_height[crouch] + relative_target[crouch]

        # Analytic derivative of minimum jerk supplies a phase-consistent target.
        down_velocity = -self.crouch_commanded_drop * 30.0 * down_p.square() * (1.0 - down_p).square() / self.crouch_down_duration.clamp_min(1.0e-6)
        return_velocity = self.crouch_commanded_drop * 30.0 * return_p.square() * (1.0 - return_p).square() / self.crouch_return_duration.clamp_min(1.0e-6)
        self.target_vertical_velocity[crouch] = 0.0
        self.target_vertical_velocity[down] = down_velocity[down]
        self.target_vertical_velocity[returning] = return_velocity[returning]
        self.crouch_phase[down], self.crouch_phase[hold] = 1, 2
        self.crouch_phase[returning], self.crouch_phase[stand_hold] = 3, 4
        self.crouch_phase_progress[down] = down_p[down]
        self.crouch_phase_progress[hold] = hold_p[hold]
        self.crouch_phase_progress[returning] = return_p[returning]
        self.crouch_phase_progress[stand_hold] = stand_p[stand_hold]
        self.crouch_hold_progress[moving] = torch.where(hold[moving], hold_p[moving], (motion_elapsed[moving] >= hold_end[moving]).float())
        self.crouch_return_progress[moving] = torch.where(returning[moving], return_p[moving], (motion_elapsed[moving] >= return_end[moving]).float())
        self.crouch_hold_complete[moving] = motion_elapsed[moving] >= hold_end[moving]
        self.crouch_return_complete[moving] = motion_elapsed[moving] >= return_end[moving]
        self.crouch_stand_hold_complete[moving] = motion_elapsed[moving] >= total[moving]
        self.crouch_current_height_error[crouch] = self.target_pelvis_height[crouch] - height[crouch]

        limits = self.robot.data.soft_joint_pos_limits.torch[:, self._crouch_joint_ids]
        position = self.robot.data.joint_pos.torch[:, self._crouch_joint_ids]
        center = 0.5 * (limits[..., 0] + limits[..., 1])
        half_range = 0.5 * (limits[..., 1] - limits[..., 0]).clamp_min(1.0e-6)
        self.joint_limit_proximity[crouch] = ((position - center).abs() / half_range).amax(dim=1)[crouch]

    def _update_path_targets(self, cos_h: torch.Tensor, sin_h: torch.Tensor) -> None:
        """Express the active RUN centerline entirely in robot/path-local coordinates."""
        run = self.skill_id == int(SkillId.RUN)
        if not run.any():
            return
        path_heading = self.path_heading_w
        tangent_x, tangent_y = torch.cos(path_heading), torch.sin(path_heading)
        normal_x, normal_y = -tangent_y, tangent_x
        offset = self.robot.data.root_pos_w.torch[:, :2] - self.path_origin_w
        along = tangent_x * offset[:, 0] + tangent_y * offset[:, 1]
        lateral = normal_x * offset[:, 0] + normal_y * offset[:, 1]
        self.path_lateral_error[run] = lateral[run]

        velocity_w = self.robot.data.root_lin_vel_w.torch[:, :2]
        self.path_forward_velocity[run] = (
            tangent_x * velocity_w[:, 0] + tangent_y * velocity_w[:, 1]
        )[run]
        self.path_lateral_velocity[run] = (
            normal_x * velocity_w[:, 0] + normal_y * velocity_w[:, 1]
        )[run]

        lookahead_along = along + float(self.cfg.path_lookahead_distance_m)
        lookahead_w_x = self.path_origin_w[:, 0] + tangent_x * lookahead_along
        lookahead_w_y = self.path_origin_w[:, 1] + tangent_y * lookahead_along
        delta_x = lookahead_w_x - self.robot.data.root_pos_w.torch[:, 0]
        delta_y = lookahead_w_y - self.robot.data.root_pos_w.torch[:, 1]
        self.path_lookahead_b[run, 0] = (cos_h * delta_x + sin_h * delta_y)[run]
        self.path_lookahead_b[run, 1] = (-sin_h * delta_x + cos_h * delta_y)[run]

        # RUN reuses otherwise inactive target-displacement and obstacle slots.
        self.target_displacement_b[run] = self.path_lookahead_b[run]
        self.obstacle_geometry_b[run, 0] = self.path_lateral_error[run]
        self.obstacle_geometry_b[run, 1] = self.path_forward_velocity[run]
        self.obstacle_geometry_b[run, 2] = self.path_lateral_velocity[run]
        self.obstacle_geometry_b[run, 3] = self.path_curvature[run]

    def _update_extra_command(self) -> None:
        current = torch.nn.functional.one_hot(self.skill_id, SKILL_COUNT).float()
        previous = torch.nn.functional.one_hot(self.previous_skill_id, SKILL_COUNT).float()
        duration = self.segment_duration.clamp_min(1.0e-6)
        elapsed = (self.segment_elapsed / duration).clamp(0.0, 1.0)
        remaining = (1.0 - elapsed).clamp(0.0, 1.0)
        phase = elapsed
        # STOP reuses the four obstacle slots, which are otherwise inactive:
        # forward speed, required deceleration, braking target, hold progress.
        stop = self.skill_id == int(SkillId.STOP)
        self.obstacle_geometry_b[stop, 0] = self.robot.data.root_lin_vel_b.torch[stop, 0]
        self.obstacle_geometry_b[stop, 1] = self.stop_required_deceleration[stop]
        self.obstacle_geometry_b[stop, 2] = self.stop_braking_target_speed[stop]
        self.obstacle_geometry_b[stop, 3] = self.stop_hold_progress[stop]
        stop_phase = torch.where(stop, self.stop_progress, phase)
        hold_phase = torch.where(stop, self.stop_hold_progress, self.recovery_mode)
        crouch = self.skill_id == int(SkillId.CROUCH)
        relative_height_target = self.target_pelvis_height - self.crouch_entry_height
        self.target_displacement_b[crouch, 0] = self.crouch_current_height_error[crouch]
        self.target_displacement_b[crouch, 1] = (
            self.robot.data.root_pos_w.torch[crouch, 2] - self.crouch_entry_height[crouch]
        )
        self.obstacle_geometry_b[crouch, 0] = self.robot.data.root_lin_vel_w.torch[crouch, 2]
        self.obstacle_geometry_b[crouch, 1] = self.joint_limit_proximity[crouch]
        self.obstacle_geometry_b[crouch, 2] = self.crouch_return_progress[crouch]
        self.obstacle_geometry_b[crouch, 3] = self.crouch_base_transition_progress[crouch]
        pelvis_command = torch.where(crouch, relative_height_target, self.target_pelvis_height)
        skill_phase = torch.where(crouch, self.crouch_phase.float() / 4.0, stop_phase)
        completion_progress = torch.where(crouch, self.crouch_hold_progress, hold_phase)
        self.extra_command = torch.cat(
            (
                current,
                previous,
                torch.sin(self.heading_error).unsqueeze(1),
                torch.cos(self.heading_error).unsqueeze(1),
                self.target_displacement_b,
                pelvis_command.unsqueeze(1),
                self.obstacle_geometry_b,
                self.target_vertical_velocity.unsqueeze(1),
                elapsed.unsqueeze(1),
                remaining.unsqueeze(1),
                skill_phase.unsqueeze(1),
                self.transition_progress.unsqueeze(1),
                completion_progress.unsqueeze(1),
                self.target_posture_rp,
            ),
            dim=1,
        )

    def _update_command(self):
        # Keep simulator/command mutation out of an accidental outer inference
        # context. PPO calls this in normal mode, so training semantics are unchanged.
        with torch.inference_mode(False):
            return self._update_command_mutable()

    def _update_command_mutable(self):
        dt = float(self._env.step_dt)
        advance_mask = ~self._just_reset
        self.segment_elapsed[advance_mask] += dt
        self._just_reset[:] = False
        self._advance_scripts()
        self._update_turn_progress()
        self.transition_progress = torch.clamp(
            self.segment_elapsed / max(float(self.cfg.transition_duration_s), 1.0e-6), 0.0, 1.0
        )
        self._update_local_targets()
        self._update_stop_targets(dt)
        self._update_crouch_targets()
        self.vel_command_b.zero_()
        self.vel_command_b[:, 0] = self.target_speed
        self.vel_command_b[:, 2] = torch.clamp(
            float(self.cfg.heading_control_stiffness) * self.heading_error,
            min=float(self.cfg.ranges.ang_vel_z[0]),
            max=float(self.cfg.ranges.ang_vel_z[1]),
        )
        stop = self.skill_id == int(SkillId.STOP)
        stop_error = self.heading_error[stop]
        dead_zone = float(self.cfg.stop_heading_dead_zone_rad)
        effective_error = torch.sign(stop_error) * torch.relu(stop_error.abs() - dead_zone)
        yaw_limit = float(self.cfg.stop_heading_yaw_rate_limit_rps)
        self.vel_command_b[stop, 2] = yaw_limit * torch.tanh(
            float(self.cfg.stop_heading_feedback_gain) * effective_error / max(yaw_limit, 1.0e-6)
        )
        self.heading_target.copy_(self.target_heading_w)
        self.is_standing_env = (self.skill_id == int(SkillId.STOP)) | (self.skill_id == int(SkillId.CROUCH))
        self._update_extra_command()

    def blend_weight(self, skill_ids: tuple[int, ...]) -> torch.Tensor:
        """Cross-faded activation for skill-specific rewards."""
        selected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        previous = torch.zeros_like(selected)
        for skill_id in skill_ids:
            selected |= self.skill_id == int(skill_id)
            previous |= self.previous_skill_id == int(skill_id)
        alpha = self.transition_progress
        return alpha * selected.float() + (1.0 - alpha) * previous.float()


def motion_command_observation(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return only appended command fields; the legacy 3-D command stays in place."""
    term = env.command_manager.get_term(command_name)
    return term.extra_command
