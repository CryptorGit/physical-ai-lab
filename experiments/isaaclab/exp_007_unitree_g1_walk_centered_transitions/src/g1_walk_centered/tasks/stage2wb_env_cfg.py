"""Stage 2W-B steady-WALK stabilization training environment."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString

from .stage2w_env_cfg import IndependentWalkEnvCfg


@configclass
class WalkStabilizationEnvCfg(IndependentWalkEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.class_type = ResolvableString(
            "g1_walk_centered.tasks.stage2wb_mdp:StabilizedWalkCommand"
        )
        self.commands.base_velocity.heading_perturb_probability = 0.5
        self.commands.base_velocity.heading_perturb_amplitude_max_rad = 0.06
        self.commands.base_velocity.heading_perturb_frequency_hz = (0.08, 0.15)
        # Start from the selected Stage 2W pilot-1 reward profile.
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.heading_error.weight = -1.0
        self.rewards.lateral_velocity.weight = -0.5
        self.rewards.cross_track_error.weight = -0.25
        self.rewards.yaw_rate_oscillation = RewTerm(
            func="g1_walk_centered.tasks.stage2wb_mdp:YawRateOscillationPenalty",
            weight=-0.02,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
