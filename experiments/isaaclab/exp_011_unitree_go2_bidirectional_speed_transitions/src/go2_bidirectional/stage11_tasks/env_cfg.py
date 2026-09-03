"""Stage 11: Stage 7 task plus one physical tangential-slip reward."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString
from isaaclab_physx.sensors import ContactSensorCfg

from go2_bidirectional.stage7_tasks.env_cfg import Exp011Go2LowSpeedEnvCfg

GROUND = "/World/ground/terrain/GroundPlane/CollisionPlane"


@configclass
class Exp011Go2TangentialSlipEnvCfg(Exp011Go2LowSpeedEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.class_type = ResolvableString(
            "go2_bidirectional.stage11_tasks.command:PhaseGatedLowSpeedVelocityCommand"
        )
        self.scene.stage11_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
            update_period=0.0,
            track_pose=True,
            track_contact_points=True,
            track_friction_forces=True,
            max_contact_data_count_per_prim=16,
            filter_prim_paths_expr=[GROUND],
        )
        self.rewards.go2_contact_tangential_slip = RewTerm(
            func=ResolvableString(
                "go2_bidirectional.stage11_tasks.reward:go2_contact_tangential_slip"
            ),
            weight=1.0,
        )
