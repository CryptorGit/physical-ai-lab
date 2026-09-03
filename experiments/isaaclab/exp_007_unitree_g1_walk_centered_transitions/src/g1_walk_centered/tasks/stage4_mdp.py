"""Stage 4 one-way WALK_TO_STAND transition training terms."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.envs.mdp.commands import UniformVelocityCommand

from g1_walk_centered.command_contract import MotionCommand
from g1_walk_centered.experts import load_walk_expert
from g1_walk_centered.experts.adapters import CanonicalRobotState
from g1_walk_centered.tasks.stage2w_mdp import minimum_jerk


SUPPORTED_SPEEDS = (0.6, 0.8, 1.0, 1.2)


class WalkToStandCommand(UniformVelocityCommand):
    """Generate actual WALK occupancy followed by a minimum-jerk stop request."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ramp_duration_s = float(getattr(cfg, "ramp_duration_s", 1.6))
        self.transition_timeout_s = float(getattr(cfg, "transition_timeout_s", 4.0))
        self.k_heading = float(getattr(cfg, "k_heading", 0.8))
        self.k_yaw_rate = float(getattr(cfg, "k_yaw_rate", 0.10))
        self.yaw_limit = float(getattr(cfg, "yaw_rate_limit", 0.30))
        self.low_pass_alpha = float(getattr(cfg, "low_pass_alpha", 0.15))
        self.slew_limit = float(getattr(cfg, "yaw_rate_slew_limit", 0.01))
        self.target_speed = torch.full((self.num_envs,), 0.6, device=self.device)
        self.target_heading_w = torch.zeros(self.num_envs, device=self.device)
        self.path_origin_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.filtered_yaw_command = torch.zeros(self.num_envs, device=self.device)
        # 0=STAND source seed, 1=STAND_TO_WALK, 2=WALK hold, 3=WALK_TO_STAND.
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.settle_streak = torch.zeros(self.num_envs, device=self.device)
        self.stand_hold_duration = torch.ones(self.num_envs, device=self.device)
        self.source_completion_streak = torch.zeros(self.num_envs, device=self.device)
        self.walk_hold_streak = torch.zeros(self.num_envs, device=self.device)
        self.walk_hold_duration = torch.full((self.num_envs,), 2.0, device=self.device)
        self.completion_streak = torch.zeros(self.num_envs, device=self.device)
        self.support_switches = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.previous_support = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.no_switch_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.source_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _contacts(self) -> torch.Tensor:
        sensor = self._env.scene.sensors["contact_forces"]
        names = self.robot.find_bodies(".*_ankle_roll_link")[1]
        ids = [sensor.body_names.index(name) for name in names]
        forces = sensor.data.net_forces_w_history.torch[:, :, ids, :]
        return forces.norm(dim=-1).amax(dim=1) > 5.0

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        speeds = torch.tensor(SUPPORTED_SPEEDS, device=self.device)
        self.target_speed[ids] = speeds[
            torch.randint(0, len(SUPPORTED_SPEEDS), (len(ids),), device=self.device)
        ]
        self.target_heading_w[ids] = self.robot.data.heading_w.torch[ids]
        self.path_origin_xy[ids] = self.robot.data.root_pos_w.torch[ids, :2]
        self.filtered_yaw_command[ids] = 0.0
        self.phase[ids] = 0
        self.phase_elapsed[ids] = 0.0
        self.settle_streak[ids] = 0.0
        self.stand_hold_duration[ids] = 0.8 + torch.rand(len(ids), device=self.device)
        self.source_completion_streak[ids] = 0.0
        self.walk_hold_streak[ids] = 0.0
        self.walk_hold_duration[ids] = 2.0 + 1.5 * torch.rand(len(ids), device=self.device)
        self.completion_streak[ids] = 0.0
        self.support_switches[ids] = 0
        self.previous_support[ids] = 0
        self.no_switch_elapsed[ids] = 0.0
        self.completed[ids] = False
        self.source_failed[ids] = False
        self.vel_command_b[ids] = 0.0
        self.is_standing_env[ids] = True

    def _update_command(self):
        dt = float(self._env.step_dt)
        contacts = self._contacts()
        support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
        g = self.robot.data.projected_gravity_b.torch
        roll = torch.atan2(g[:, 1], -g[:, 2])
        pitch = torch.atan2(-g[:, 0], torch.sqrt(g[:, 1] ** 2 + g[:, 2] ** 2))
        horizontal = self.robot.data.root_lin_vel_b.torch[:, :2].norm(dim=1)
        vertical = self.robot.data.root_lin_vel_w.torch[:, 2].abs()
        vx = self.robot.data.root_lin_vel_b.torch[:, 0]
        error = torch.atan2(
            torch.sin(self.target_heading_w - self.robot.data.heading_w.torch),
            torch.cos(self.target_heading_w - self.robot.data.heading_w.torch),
        )
        phase_before = self.phase.clone()
        stand = phase_before == 0
        source_edge = phase_before == 1
        walk = phase_before == 2
        edge = phase_before == 3

        safe_stand = (
            (horizontal <= 0.08)
            & (vertical <= 0.05)
            & (roll.abs() <= 0.10)
            & (pitch.abs() <= 0.10)
            & contacts.all(dim=1)
        )
        self.settle_streak[:] = torch.where(
            stand & safe_stand, self.settle_streak + dt, torch.zeros_like(self.settle_streak)
        )
        start_source = stand & (self.settle_streak >= 0.4) & (
            self.phase_elapsed >= self.stand_hold_duration
        )
        self.phase[start_source] = 1
        self.phase_elapsed[start_source] = 0.0
        self.target_heading_w[start_source] = self.robot.data.heading_w.torch[start_source]
        self.support_switches[start_source] = 0
        self.previous_support[start_source] = support[start_source]

        switched_source = source_edge & (support != self.previous_support) & (support != 0)
        self.support_switches += switched_source.long()
        self.previous_support[:] = torch.where(source_edge, support, self.previous_support)
        source_good = (
            source_edge
            & (vx >= 0.75 * self.target_speed)
            & ((vx - self.target_speed).abs() <= 0.20)
            & (error.abs() <= 0.12)
            & (roll.abs() <= 0.20)
            & (pitch.abs() <= 0.20)
            & (self.support_switches >= 2)
        )
        self.source_completion_streak[:] = torch.where(
            source_good,
            self.source_completion_streak + dt,
            torch.zeros_like(self.source_completion_streak),
        )
        start_walk = source_edge & (self.source_completion_streak >= 0.4)
        self.phase[start_walk] = 2
        self.phase_elapsed[start_walk] = 0.0
        self.walk_hold_streak[start_walk] = 0.0

        walk_good = walk & ((vx - self.target_speed).abs() <= 0.20) & (error.abs() <= 0.12)
        self.walk_hold_streak[:] = torch.where(
            walk_good,
            self.walk_hold_streak + dt,
            torch.zeros_like(self.walk_hold_streak),
        )
        start_edge = walk & (self.walk_hold_streak >= self.walk_hold_duration)
        self.phase[start_edge] = 3
        self.phase_elapsed[start_edge] = 0.0
        self.path_origin_xy[start_edge] = self.robot.data.root_pos_w.torch[start_edge, :2]
        self.previous_support[start_edge] = support[start_edge]
        self.no_switch_elapsed[start_edge] = 0.0

        switched_edge = edge & (support != self.previous_support) & (support != 0)
        self.no_switch_elapsed[:] = torch.where(
            edge,
            torch.where(
                switched_edge,
                torch.zeros_like(self.no_switch_elapsed),
                self.no_switch_elapsed + dt,
            ),
            self.no_switch_elapsed,
        )
        self.previous_support[:] = torch.where(edge, support, self.previous_support)
        completion_good = (
            edge
            & (horizontal <= 0.08)
            & (vertical <= 0.05)
            & (error.abs() <= 0.12)
            & (roll.abs() <= 0.10)
            & (pitch.abs() <= 0.10)
            & contacts.all(dim=1)
            & (self.no_switch_elapsed >= 0.4)
        )
        self.completion_streak[:] = torch.where(
            completion_good,
            self.completion_streak + dt,
            torch.zeros_like(self.completion_streak),
        )
        self.completed |= self.completion_streak >= 0.4

        self.source_failed |= (
            (stand & (self.phase_elapsed >= 2.0))
            | (source_edge & (self.phase_elapsed >= 4.0))
            | (walk & (self.phase_elapsed >= 6.0))
        )
        self.phase_elapsed += dt
        self.is_standing_env[:] = stand
        source_ramp = self.target_speed * minimum_jerk(self.phase_elapsed / 1.5)
        stop_ramp = self.target_speed * (
            1.0 - minimum_jerk(self.phase_elapsed / self.ramp_duration_s)
        )
        command_vx = torch.where(
            source_edge,
            source_ramp,
            torch.where(walk, self.target_speed, torch.where(edge, stop_ramp, 0.0)),
        )
        raw = (
            self.k_heading * error - self.k_yaw_rate * self.robot.data.root_ang_vel_b.torch[:, 2]
        ).clamp(-self.yaw_limit, self.yaw_limit)
        low_pass = self.filtered_yaw_command + self.low_pass_alpha * (
            raw - self.filtered_yaw_command
        )
        self.filtered_yaw_command += (
            low_pass - self.filtered_yaw_command
        ).clamp(-self.slew_limit, self.slew_limit)
        self.filtered_yaw_command[stand] = 0.0
        self.vel_command_b.zero_()
        self.vel_command_b[:, 0] = command_vx
        self.vel_command_b[:, 2] = torch.where(
            stand, torch.zeros_like(command_vx), self.filtered_yaw_command
        )


class RoutedWalkToStandJointPositionAction(JointPositionAction):
    """Route frozen source controllers and apply only transition actions on the edge."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.stand_expert = load_walk_expert(cfg.stand_checkpoint_path, device=self.device)
        self.walk_expert = load_walk_expert(cfg.walk_checkpoint_path, device=self.device)
        self.stand_to_walk_expert = load_walk_expert(
            cfg.stand_to_walk_checkpoint_path, device=self.device
        )
        self.policy_actions = torch.zeros_like(self._raw_actions)
        self.stand_actions = torch.zeros_like(self._raw_actions)
        self.walk_actions = torch.zeros_like(self._raw_actions)
        self.start_actions = torch.zeros_like(self._raw_actions)

    def _state(self) -> CanonicalRobotState:
        robot = self._asset
        return CanonicalRobotState(
            robot.data.root_lin_vel_b.torch,
            robot.data.root_ang_vel_b.torch,
            robot.data.projected_gravity_b.torch,
            robot.data.heading_w.torch,
            robot.data.joint_pos.torch - robot.data.default_joint_pos.torch,
            robot.data.joint_vel.torch - robot.data.default_joint_vel.torch,
            self._raw_actions,
        )

    def process_actions(self, actions: torch.Tensor):
        term = self._env.command_manager.get_term("base_velocity")
        state = self._state()
        zeros = torch.zeros(self.num_envs, device=self.device)
        with torch.inference_mode():
            self.stand_actions[:] = self.stand_expert(
                state, MotionCommand(zeros, term.target_heading_w, target_yaw_rate_radps=zeros)
            )
            command = MotionCommand(
                term.target_speed,
                term.target_heading_w,
                target_yaw_rate_radps=term.filtered_yaw_command,
            )
            self.walk_actions[:] = self.walk_expert(state, command)
            self.start_actions[:] = self.stand_to_walk_expert(state, command)
        self.policy_actions[:] = actions
        effective = torch.where(
            (term.phase == 0).unsqueeze(1),
            self.stand_actions,
            torch.where(
                (term.phase == 1).unsqueeze(1),
                self.start_actions,
                torch.where((term.phase == 2).unsqueeze(1), self.walk_actions, actions),
            ),
        )
        super().process_actions(effective)


def transition_completed(env, command_name: str) -> torch.Tensor:
    return env.command_manager.get_term(command_name).completed


def transition_failed(env, command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    return term.source_failed | (
        (term.phase == 3) & (term.phase_elapsed >= term.transition_timeout_s)
    )


def completion_bonus(env, command_name: str) -> torch.Tensor:
    return env.command_manager.get_term(command_name).completed.float()


def endpoint_alignment(env, command_name: str, boundary: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    action = env.action_manager.get_term("joint_pos")
    if boundary == "source":
        active = (term.phase == 3) & (term.phase_elapsed <= 0.30)
        reference = action.walk_actions
    elif boundary == "target":
        active = (term.phase == 3) & (term.completion_streak > 0.0)
        reference = action.stand_actions
    else:
        raise ValueError(f"unknown boundary: {boundary}")
    return (action.policy_actions - reference).square().mean(dim=1) * active


def transition_masked_heading_error(env, command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    error = torch.atan2(
        torch.sin(term.target_heading_w - term.robot.data.heading_w.torch),
        torch.cos(term.target_heading_w - term.robot.data.heading_w.torch),
    )
    return error.square() * (term.phase == 3)


def transition_lateral_velocity(env, command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    return term.robot.data.root_lin_vel_b.torch[:, 1].square() * (term.phase == 3)


def reverse_motion(env, command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    return torch.relu(-term.robot.data.root_lin_vel_b.torch[:, 0] - 0.10).square() * (
        term.phase == 3
    )


def double_support_progress(env, command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    return term._contacts().all(dim=1).float() * (term.phase == 3)
