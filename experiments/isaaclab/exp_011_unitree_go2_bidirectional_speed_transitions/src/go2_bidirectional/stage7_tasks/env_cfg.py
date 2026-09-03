"""Stage 4 environment with only the Stage 7 command term substituted."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString
from go2_bidirectional.stage2_tasks.env_cfg import Exp011Go2BidirectionalEnvCfg


@configclass
class Exp011Go2LowSpeedEnvCfg(Exp011Go2BidirectionalEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.class_type = ResolvableString(
            "go2_bidirectional.stage7_tasks.command:LowSpeedGaitVelocityCommand"
        )
