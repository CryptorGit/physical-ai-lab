"""One-way canonical-state adapters for the immutable experts."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from g1_walk_centered.command_contract import MotionCommand


LEGACY_OBSERVATION_DIM = 123
COMMAND_OBSERVATION_DIM = 29
RUN_OBSERVATION_DIM = 152
ACTION_DIM = 37
FORBIDDEN_INPUTS = frozenset({"absolute_world_x", "absolute_world_y"})


@dataclass(frozen=True)
class CanonicalRobotState:
    """Policy-relative state; deliberately excludes absolute world XY."""

    base_linear_velocity_body_mps: torch.Tensor
    base_angular_velocity_body_radps: torch.Tensor
    projected_gravity_body: torch.Tensor
    heading_w_rad: torch.Tensor
    joint_position_relative: torch.Tensor
    joint_velocity_relative: torch.Tensor
    previous_action: torch.Tensor

    def validate(self) -> None:
        expected = (3, 3, 3, ACTION_DIM, ACTION_DIM, ACTION_DIM)
        values = (
            self.base_linear_velocity_body_mps,
            self.base_angular_velocity_body_radps,
            self.projected_gravity_body,
            self.joint_position_relative,
            self.joint_velocity_relative,
            self.previous_action,
        )
        batch = values[0].shape[:-1]
        for value, width in zip(values, expected, strict=True):
            if value.shape[:-1] != batch or value.shape[-1] != width:
                raise ValueError(f"canonical state shape mismatch: expected (*,{width}), got {tuple(value.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError("canonical state contains non-finite values")
        if self.heading_w_rad.shape != batch or not torch.isfinite(self.heading_w_rad).all():
            raise ValueError(f"heading_w_rad must have shape {batch}")


def nominal_state(batch_size: int = 1, *, device: str | torch.device = "cpu") -> CanonicalRobotState:
    zeros3 = torch.zeros(batch_size, 3, device=device, dtype=torch.float32)
    zeros37 = torch.zeros(batch_size, ACTION_DIM, device=device, dtype=torch.float32)
    gravity = zeros3.clone()
    gravity[:, 2] = -1.0
    heading = torch.zeros(batch_size, device=device, dtype=torch.float32)
    return CanonicalRobotState(zeros3, zeros3.clone(), gravity, heading, zeros37, zeros37.clone(), zeros37.clone())


def canonical_state_from_legacy_observation(
    observation: torch.Tensor,
    *,
    heading_w_rad: torch.Tensor | None = None,
) -> CanonicalRobotState:
    """Decode the canonical state from an immutable expert's 123-D observation.

    Command columns 9:12 are deliberately discarded.  Callers must supply a
    validated :class:`MotionCommand`, which is the only way commands re-enter
    the expert-specific observation.
    """
    if observation.shape[-1] != LEGACY_OBSERVATION_DIM:
        raise ValueError(f"expected 123-D legacy observation, got {tuple(observation.shape)}")
    if not torch.isfinite(observation).all():
        raise ValueError("legacy observation contains non-finite values")
    batch = observation.shape[:-1]
    heading = (
        torch.zeros(batch, dtype=observation.dtype, device=observation.device)
        if heading_w_rad is None
        else heading_w_rad.to(dtype=observation.dtype, device=observation.device)
    )
    return CanonicalRobotState(
        base_linear_velocity_body_mps=observation[..., 0:3],
        base_angular_velocity_body_radps=observation[..., 3:6],
        projected_gravity_body=observation[..., 6:9],
        heading_w_rad=heading,
        joint_position_relative=observation[..., 12:49],
        joint_velocity_relative=observation[..., 49:86],
        previous_action=observation[..., 86:123],
    )


def _legacy_observation(state: CanonicalRobotState, command: MotionCommand) -> torch.Tensor:
    state.validate()
    first = state.base_linear_velocity_body_mps
    command_columns = torch.zeros((*first.shape[:-1], 3), dtype=first.dtype, device=first.device)
    speed = command.target_speed_mps
    yaw_rate = command.target_yaw_rate_radps
    command_columns[..., 0] = (
        speed.to(dtype=first.dtype, device=first.device)
        if isinstance(speed, torch.Tensor) else float(speed)
    )
    command_columns[..., 2] = (
        yaw_rate.to(dtype=first.dtype, device=first.device)
        if isinstance(yaw_rate, torch.Tensor) else float(yaw_rate)
    )
    result = torch.cat(
        (
            state.base_linear_velocity_body_mps,
            state.base_angular_velocity_body_radps,
            state.projected_gravity_body,
            command_columns,
            state.joint_position_relative,
            state.joint_velocity_relative,
            state.previous_action,
        ),
        dim=-1,
    )
    if result.shape[-1] != LEGACY_OBSERVATION_DIM:
        raise AssertionError("legacy observation construction failed")
    return result


def to_walk_observation(state: CanonicalRobotState, command: MotionCommand) -> torch.Tensor:
    """Build the exact 123-D exp_005 observation."""
    return _legacy_observation(state, command)


def to_run_observation(
    state: CanonicalRobotState,
    command: MotionCommand,
    *,
    route: str = "RUN",
) -> torch.Tensor:
    """Build the exact 152-D exp_006 RUN/TURN observation."""
    if route not in {"RUN", "TURN"}:
        raise ValueError("Stage 0 run adapter exposes only RUN and TURN")
    legacy = _legacy_observation(state, command)
    extra = torch.zeros((*legacy.shape[:-1], COMMAND_OBSERVATION_DIM), dtype=legacy.dtype, device=legacy.device)
    skill = 0 if route == "RUN" else 2
    extra[..., skill] = 1.0
    extra[..., 6 + skill] = 1.0
    target_heading = torch.as_tensor(command.target_heading_w_rad, dtype=legacy.dtype, device=legacy.device)
    heading_error = torch.atan2(
        torch.sin(target_heading - state.heading_w_rad),
        torch.cos(target_heading - state.heading_w_rad),
    )
    extra[..., 12] = torch.sin(heading_error)
    extra[..., 13] = torch.cos(heading_error)
    extra[..., 14] = 1.0 if route == "RUN" else heading_error
    extra[..., 23] = 1.0
    extra[..., 25] = 1.0
    result = torch.cat((legacy, extra), dim=-1)
    if result.shape[-1] != RUN_OBSERVATION_DIM or not torch.isfinite(result).all():
        raise AssertionError("run observation construction failed")
    return result
