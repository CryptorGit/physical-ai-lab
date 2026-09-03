"""Independent steady-WALK training and evaluation environments."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString

from g1_flat_run.tasks.g1_flat_run_env_cfg import G1FlatRunStage2EnvCfg


@configclass
class IndependentWalkEnvCfg(G1FlatRunStage2EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 10.0
        self.commands.base_velocity.class_type = ResolvableString(
            "g1_walk_centered.tasks.stage2w_mdp:IndependentWalkCommand"
        )
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ramp_duration_s = 1.5
        self.commands.base_velocity.k_heading = 0.8
        self.commands.base_velocity.k_yaw_rate = 0.10
        self.commands.base_velocity.yaw_rate_limit = 0.30
        self.commands.base_velocity.low_pass_alpha = 0.15
        self.commands.base_velocity.yaw_rate_slew_limit = 0.01

        robot = SceneEntityCfg("robot")
        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.heading_error = RewTerm(
            func="g1_walk_centered.tasks.stage2w_mdp:heading_error_l2",
            weight=-2.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.lateral_velocity = RewTerm(
            func="g1_walk_centered.tasks.stage2w_mdp:lateral_velocity_l2",
            weight=-0.75,
            params={"asset_cfg": robot},
        )
        self.rewards.cross_track_error = RewTerm(
            func="g1_walk_centered.tasks.stage2w_mdp:cross_track_error_l2",
            weight=-0.5,
            params={"command_name": "base_velocity"},
        )
        self.rewards.ankle_pitch_effort_hinge = RewTerm(
            func="g1_walk_centered.tasks.stage2w_mdp:AnklePitchEffortHinge",
            weight=-0.25,
            params={"asset_cfg": robot, "threshold": 0.95},
        )
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.dof_pos_limits.weight = -0.1
        self.rewards.feet_slide.weight = -0.25


@configclass
class IndependentWalkEvalEnvCfg(IndependentWalkEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
