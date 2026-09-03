"""Stage 7 command sampling and mild high-speed-only slip shaping."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import SceneEntityCfg

from .stage3_mdp import FocusedRunVelocityCommand
from .stage6_mdp import Stage6VelocityCommand


class Stage7VelocityCommand(Stage6VelocityCommand):
    """Focus 70% on 4.40--4.50 m/s and expose 4.55 m/s sparingly."""

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        # Bypass the earlier focused samplers, retaining the official lateral,
        # yaw and standing-mask sampling before replacing forward velocity.
        super(FocusedRunVelocityCommand, self)._resample_command(env_ids)
        count = len(env_ids)
        if count == 0:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        draw = torch.rand(count, device=self.device)
        focused = torch.empty(count, device=self.device).uniform_(4.40, 4.50)
        full = torch.empty(count, device=self.device).uniform_(4.25, 4.55)
        upper_probe = torch.empty(count, device=self.device).uniform_(4.50, 4.55)
        self.vel_command_b[ids, 0] = torch.where(
            draw < 0.70,
            focused,
            torch.where(draw < 0.90, full, upper_probe),
        )


def high_speed_feet_slide(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    start_speed_mps: float = 4.40,
    full_speed_mps: float = 4.50,
) -> torch.Tensor:
    """Add a gradually ramped slip cost only above 4.40 m/s.

    The inherited -0.25 feet-slide term remains unchanged.  With Stage 7's
    -0.05 weight this adds 0 at 4.40, -0.025 at 4.45 and -0.05 at 4.50+.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .amax(dim=1)
        > 1.0
    )
    asset = env.scene[asset_cfg.name]
    foot_speed = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2].norm(dim=-1)
    slide = torch.sum(foot_speed * contacts, dim=1)
    command_speed = env.command_manager.get_command(command_name)[:, 0]
    ramp = torch.clamp(
        (command_speed - start_speed_mps) / max(full_speed_mps - start_speed_mps, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    return slide * ramp
