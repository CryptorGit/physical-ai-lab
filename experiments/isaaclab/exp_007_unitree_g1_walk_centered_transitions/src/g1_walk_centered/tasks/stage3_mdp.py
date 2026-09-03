"""Stage 3 one-way STAND_TO_WALK transition terms.

The action term enforces the runtime ownership contract during training:
the frozen STAND actor owns source-state generation and only the trainable
transition action reaches the robot after the transition request.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from g1_walk_centered.command_contract import MotionCommand
from g1_walk_centered.experts import load_walk_expert
from g1_walk_centered.experts.adapters import CanonicalRobotState
from g1_walk_centered.tasks.stage2w_mdp import minimum_jerk


SUPPORTED_SPEEDS = (0.6, 0.8, 1.0, 1.2)


class StandToWalkCommand(UniformVelocityCommand):
    """State-conditioned source hold and minimum-jerk transition command."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ramp_duration_s = float(getattr(cfg, "ramp_duration_s", 1.5))
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
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 0=STAND, 1=EDGE
        self.settle_streak = torch.zeros(self.num_envs, device=self.device)
        self.stand_hold_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.stand_hold_duration = torch.ones(self.num_envs, device=self.device)
        self.transition_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.completion_streak = torch.zeros(self.num_envs, device=self.device)
        self.support_switches = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.previous_support = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
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
        self.target_speed[ids] = speeds[torch.randint(0, len(SUPPORTED_SPEEDS), (len(ids),), device=self.device)]
        self.target_heading_w[ids] = self.robot.data.heading_w.torch[ids]
        self.path_origin_xy[ids] = self.robot.data.root_pos_w.torch[ids, :2]
        self.filtered_yaw_command[ids] = 0.0
        self.phase[ids] = 0
        self.settle_streak[ids] = 0.0
        self.stand_hold_elapsed[ids] = 0.0
        self.stand_hold_duration[ids] = 0.8 + torch.rand(len(ids), device=self.device)
        self.transition_elapsed[ids] = 0.0
        self.completion_streak[ids] = 0.0
        self.support_switches[ids] = 0
        self.previous_support[ids] = 0
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
        safe_stand = (
            (horizontal <= 0.08)
            & (vertical <= 0.05)
            & (roll.abs() <= 0.10)
            & (pitch.abs() <= 0.10)
            & contacts.all(dim=1)
        )
        stand = self.phase == 0
        self.settle_streak[:] = torch.where(stand & safe_stand, self.settle_streak + dt, torch.zeros_like(self.settle_streak))
        settled = self.settle_streak >= 0.4
        self.stand_hold_elapsed[:] = torch.where(
            stand & settled, self.stand_hold_elapsed + dt, self.stand_hold_elapsed
        )
        begin = stand & settled & (self.stand_hold_elapsed >= self.stand_hold_duration)
        self.phase[begin] = 1
        self.transition_elapsed[begin] = 0.0
        self.target_heading_w[begin] = self.robot.data.heading_w.torch[begin]
        self.path_origin_xy[begin] = self.robot.data.root_pos_w.torch[begin, :2]
        self.previous_support[begin] = support[begin]
        self.is_standing_env[:] = self.phase == 0
        self.source_failed |= stand & (self._env.episode_length_buf.float() * dt > 3.8)

        edge = self.phase == 1
        self.transition_elapsed[:] = torch.where(edge, self.transition_elapsed + dt, self.transition_elapsed)
        vx = self.target_speed * minimum_jerk(self.transition_elapsed / self.ramp_duration_s)
        error = torch.atan2(
            torch.sin(self.target_heading_w - self.robot.data.heading_w.torch),
            torch.cos(self.target_heading_w - self.robot.data.heading_w.torch),
        )
        raw = (self.k_heading * error - self.k_yaw_rate * self.robot.data.root_ang_vel_b.torch[:, 2]).clamp(
            -self.yaw_limit, self.yaw_limit
        )
        low_pass = self.filtered_yaw_command + self.low_pass_alpha * (raw - self.filtered_yaw_command)
        self.filtered_yaw_command += (low_pass - self.filtered_yaw_command).clamp(-self.slew_limit, self.slew_limit)
        self.filtered_yaw_command[stand] = 0.0
        self.vel_command_b.zero_()
        self.vel_command_b[:, 0] = torch.where(edge, vx, torch.zeros_like(vx))
        self.vel_command_b[:, 2] = torch.where(edge, self.filtered_yaw_command, torch.zeros_like(vx))

        switched = edge & (support != self.previous_support) & (support != 0)
        self.support_switches += switched.long()
        self.previous_support[:] = torch.where(edge, support, self.previous_support)
        speed = self.robot.data.root_lin_vel_b.torch[:, 0]
        good = (
            edge
            & (speed >= 0.75 * self.target_speed)
            & ((speed - self.target_speed).abs() <= 0.20)
            & (error.abs() <= 0.12)
            & (roll.abs() <= 0.20)
            & (pitch.abs() <= 0.20)
            & (self.support_switches >= 2)
        )
        self.completion_streak[:] = torch.where(good, self.completion_streak + dt, torch.zeros_like(self.completion_streak))
        self.completed |= self.completion_streak >= 0.4


class RoutedTransitionJointPositionAction(JointPositionAction):
    """Apply frozen STAND actions before request and transition actions after it."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.stand_expert = load_walk_expert(cfg.stand_checkpoint_path, device=self.device)
        self.walk_expert = load_walk_expert(cfg.walk_checkpoint_path, device=self.device)
        self.policy_actions = torch.zeros_like(self._raw_actions)
        self.stand_actions = torch.zeros_like(self._raw_actions)
        self.walk_actions = torch.zeros_like(self._raw_actions)

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
            self.walk_actions[:] = self.walk_expert(
                state,
                MotionCommand(
                    term.target_speed,
                    term.target_heading_w,
                    target_yaw_rate_radps=term.filtered_yaw_command,
                ),
            )
        self.policy_actions[:] = actions
        effective = torch.where((term.phase == 0).unsqueeze(1), self.stand_actions, actions)
        super().process_actions(effective)


def transition_completed(env, command_name: str) -> torch.Tensor:
    return env.command_manager.get_term(command_name).completed


def transition_failed(env, command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    return term.source_failed | ((term.phase == 1) & (term.transition_elapsed >= term.transition_timeout_s))


def completion_bonus(env, command_name: str) -> torch.Tensor:
    return env.command_manager.get_term(command_name).completed.float()


def endpoint_alignment(env, command_name: str, boundary: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    action = env.action_manager.get_term("joint_pos")
    if boundary == "source":
        active = (term.phase == 1) & (term.transition_elapsed <= 0.30)
        reference = action.stand_actions
    elif boundary == "target":
        active = (term.phase == 1) & (term.completion_streak > 0.0)
        reference = action.walk_actions
    else:
        raise ValueError(f"unknown boundary: {boundary}")
    return (action.policy_actions - reference).square().mean(dim=1) * active


def transition_masked_heading_error(env, command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    error = torch.atan2(
        torch.sin(term.target_heading_w - term.robot.data.heading_w.torch),
        torch.cos(term.target_heading_w - term.robot.data.heading_w.torch),
    )
    return error.square() * (term.phase == 1)


def transition_lateral_velocity(env, command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    return term.robot.data.root_lin_vel_b.torch[:, 1].square() * (term.phase == 1)
