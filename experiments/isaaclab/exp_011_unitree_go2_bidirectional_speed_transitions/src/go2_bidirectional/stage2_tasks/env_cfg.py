"""Official Go2 flat task with only its command distribution replaced."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg


@configclass
class Exp011Go2BidirectionalEnvCfg(UnitreeGo2FlatEnvCfg):
    """Reward/observation/action/physics-identical Stage 2 training task."""

    def __post_init__(self) -> None:
        super().__post_init__()
        command = self.commands.base_velocity
        command.class_type = ResolvableString(
            "go2_bidirectional.stage2_tasks.command:SymmetricBidirectionalVelocityCommand"
        )
        command.resampling_time_range = (1.0e9, 1.0e9)
        command.heading_command = False
        command.rel_heading_envs = 0.0
        command.rel_standing_envs = 0.0
        command.ranges.lin_vel_x = (0.0, 2.0)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.ranges.heading = None
