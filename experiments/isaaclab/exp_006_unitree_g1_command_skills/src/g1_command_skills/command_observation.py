"""Canonical legacy/new command layout and coherent command interventions."""

from __future__ import annotations

import math

import torch


LEGACY_OBSERVATION_DIM = 123
EXTRA_COMMAND_DIM = 29
LEGACY_COMMAND_SLICE = slice(9, 12)
NEW_COMMAND_SLICE = slice(LEGACY_OBSERVATION_DIM, LEGACY_OBSERVATION_DIM + EXTRA_COMMAND_DIM)

LEGACY_LAYOUT = (
    (0, 3, "base_linear_velocity_body_mps"),
    (3, 6, "base_angular_velocity_body_radps"),
    (6, 9, "projected_gravity_body"),
    (9, 12, "generated_velocity_command_body_[vx,vy,yaw_rate]"),
    (12, 49, "joint_position_relative_37"),
    (49, 86, "joint_velocity_relative_37"),
    (86, 123, "previous_action_37"),
)

NEW_COMMAND_LAYOUT = (
    (0, 6, "current_skill_one_hot"),
    (6, 12, "previous_skill_one_hot"),
    (12, 13, "sin_target_heading_error"),
    (13, 14, "cos_target_heading_error"),
    (14, 16, "skill_local_target_state"),
    (16, 17, "relative_target_pelvis_height_for_crouch"),
    (17, 21, "skill_local_auxiliary_state"),
    (21, 22, "target_vertical_velocity"),
    (22, 23, "normalized_elapsed_time"),
    (23, 24, "normalized_remaining_time"),
    (24, 25, "skill_phase"),
    (25, 26, "transition_progress"),
    (26, 27, "recovery_mode"),
    (27, 29, "target_posture_roll_pitch"),
)

RUN, STOP, TURN, CROUCH, STEP_OVER = 0, 1, 2, 3, 4


def _wrap_to_pi(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def coherent_turn_observation(
    observations,
    angle_rad: float,
    *,
    speed_mps: float = 1.1,
    yaw_rate_limit_radps: float = 0.75,
    heading_control_stiffness: float = 1.5,
):
    """Change every policy-visible TURN field while preserving robot state."""
    variant = observations.clone()
    policy = variant["policy"]
    command = policy[..., NEW_COMMAND_SLICE]
    command.zero_()
    angle = torch.full_like(policy[..., 9], float(angle_rad))
    heading_error = _wrap_to_pi(angle)
    policy[..., LEGACY_COMMAND_SLICE] = 0.0
    policy[..., 9] = float(speed_mps)
    policy[..., 11] = torch.clamp(
        float(heading_control_stiffness) * heading_error,
        -float(yaw_rate_limit_radps),
        float(yaw_rate_limit_radps),
    )
    command[..., TURN] = 1.0
    command[..., 6 + TURN] = 1.0
    command[..., 12] = torch.sin(heading_error)
    command[..., 13] = torch.cos(heading_error)
    command[..., 14] = angle
    command[..., 15] = 0.0
    command[..., 23] = 1.0
    command[..., 25] = 1.0
    return variant


def coherent_run_observation(observations, *, speed_mps: float = 2.2, lateral_error_m: float = 0.0):
    """Create a complete RUN command at the same physical robot state."""
    variant = observations.clone()
    policy = variant["policy"]
    command = policy[..., NEW_COMMAND_SLICE]
    command.zero_()
    policy[..., LEGACY_COMMAND_SLICE] = 0.0
    policy[..., 9] = float(speed_mps)
    command[..., RUN] = 1.0
    command[..., 6 + RUN] = 1.0
    command[..., 13] = 1.0
    command[..., 14] = 1.0
    command[..., 17] = float(lateral_error_m)
    command[..., 23] = 1.0
    command[..., 25] = 1.0
    return variant


def coherent_stop_observation(observations, *, distance_m: float = 1.0):
    variant = observations.clone()
    policy = variant["policy"]
    command = policy[..., NEW_COMMAND_SLICE]
    command.zero_()
    policy[..., LEGACY_COMMAND_SLICE] = 0.0
    command[..., STOP] = 1.0
    command[..., 6 + STOP] = 1.0
    command[..., 13] = 1.0
    command[..., 14] = float(distance_m)
    command[..., 23] = 1.0
    command[..., 25] = 1.0
    return variant


def coherent_crouch_observation(
    observations,
    *,
    height_drop_m: float = 0.12,
    phase: float = 0.0,
    height_error_m: float = 0.0,
    target_vertical_velocity_mps: float = 0.0,
    hold_progress: float = 0.0,
    return_progress: float = 0.0,
):
    """Create a relative CROUCH command without exposing world XY coordinates."""
    variant = observations.clone()
    policy = variant["policy"]
    command = policy[..., NEW_COMMAND_SLICE]
    command.zero_()
    policy[..., LEGACY_COMMAND_SLICE] = 0.0
    command[..., CROUCH] = 1.0
    command[..., 6 + CROUCH] = 1.0
    command[..., 13] = 1.0
    command[..., 14] = float(height_error_m)
    command[..., 16] = -float(height_drop_m)
    command[..., 19] = float(return_progress)
    command[..., 20] = 1.0  # standing-base option fully selected
    command[..., 21] = float(target_vertical_velocity_mps)
    command[..., 24] = float(phase)
    command[..., 25] = 1.0
    command[..., 26] = float(hold_progress)
    return variant


def coherent_step_over_observation(
    observations, *, obstacle_distance_m: float = 0.25, obstacle_height_m: float = 0.05,
    obstacle_depth_m: float = 0.06, lead_foot: str = "left", phase: int = 0,
    phase_progress: float = 0.0, recovery_progress: float = 0.0,
):
    """Create the local STEP_OVER command contract at an identical robot state."""
    variant = observations.clone(); policy = variant["policy"]
    command = policy[..., NEW_COMMAND_SLICE]; command.zero_(); policy[..., LEGACY_COMMAND_SLICE] = 0.0
    command[..., STEP_OVER] = 1.0; command[..., 6 + STEP_OVER] = 1.0; command[..., 13] = 1.0
    command[..., 14] = float(obstacle_distance_m); command[..., 15] = 1.0 if lead_foot == "left" else -1.0
    command[..., 17] = float(obstacle_height_m); command[..., 18] = float(obstacle_depth_m)
    command[..., 19] = float(obstacle_height_m + 0.02); command[..., 20] = 1.0
    command[..., 22] = float(phase_progress); command[..., 23] = float(recovery_progress)
    command[..., 24] = float(phase) / 10.0; command[..., 25] = 1.0
    return variant


def apply_command_ablation(observations, mode: str, term=None):
    """Apply explicitly scoped command ablations to a cloned observation.

    ``shuffle`` is coherent: TURN angle/direction and its derived legacy yaw
    command are changed together.  It therefore also works with one env, unlike
    rolling columns or rolling a batch of size one.
    """
    if mode == "normal":
        return observations
    variant = observations.clone()
    policy = variant["policy"]
    command = policy[..., NEW_COMMAND_SLICE]
    if mode in ("zero", "new_command_zero"):
        command.zero_()
        return variant
    if mode == "legacy_command_zero":
        policy[..., LEGACY_COMMAND_SLICE] = 0.0
        return variant
    if mode == "all_command_zero":
        policy[..., LEGACY_COMMAND_SLICE] = 0.0
        command.zero_()
        return variant
    if mode != "shuffle":
        raise ValueError(f"Unknown command ablation: {mode}")
    if term is None:
        raise ValueError("Coherent shuffle requires the MotionCommand term")

    skill = term.skill_id
    turn_mask = skill == TURN
    if turn_mask.any():
        original = term.commanded_turn_angle_rad
        magnitude = torch.abs(original)
        alternate_magnitude = torch.where(
            magnitude < math.radians(67.5),
            torch.full_like(magnitude, math.pi / 2.0),
            torch.full_like(magnitude, math.pi / 4.0),
        )
        alternate = -torch.sign(original).where(original != 0.0, torch.ones_like(original)) * alternate_magnitude
        alternate_error = _wrap_to_pi(alternate - term.actual_accumulated_yaw_rad)
        policy[turn_mask, 9:12] = 0.0
        policy[turn_mask, 9] = term.target_speed[turn_mask]
        policy[turn_mask, 11] = torch.clamp(
            float(term.cfg.heading_control_stiffness) * alternate_error[turn_mask],
            min=float(term.cfg.ranges.ang_vel_z[0]),
            max=float(term.cfg.ranges.ang_vel_z[1]),
        )
        command[turn_mask, 12] = torch.sin(alternate_error[turn_mask])
        command[turn_mask, 13] = torch.cos(alternate_error[turn_mask])
        command[turn_mask, 14] = alternate[turn_mask]
        # Accumulated yaw is state-derived and deliberately remains unchanged.
    run_mask = skill == RUN
    if run_mask.any():
        current_speed = term.target_speed[run_mask]
        alternate_speed = torch.where(current_speed < 1.9, torch.full_like(current_speed, 2.6), torch.full_like(current_speed, 1.2))
        policy[run_mask, 9] = alternate_speed
    return variant


def changed_columns(left: torch.Tensor, right: torch.Tensor, offset: int = 0) -> list[dict]:
    delta = (left - right).abs().amax(dim=0)
    indices = (delta > 1.0e-9).nonzero(as_tuple=False).flatten().tolist()
    return [
        {
            "index": int(index + offset),
            "left": float(left[0, index].item()),
            "right": float(right[0, index].item()),
            "abs_difference": float(delta[index].item()),
        }
        for index in indices
    ]
