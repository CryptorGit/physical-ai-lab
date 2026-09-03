"""Stage 2R environments derived without modifying Isaac Lab or exp_005."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString

from g1_flat_run.tasks.g1_flat_run_env_cfg import G1FlatRunStage2EnvCfg


@configclass
class UnifiedStandWalkBaseEnvCfg(G1FlatRunStage2EnvCfg):
    """Shared 123-observation/37-action unified expert configuration."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 12.0
        self.commands.base_velocity.class_type = ResolvableString(
            "g1_walk_centered.tasks.stage2r_mdp:UnifiedStandWalkCommand"
        )
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.stage2r_phase = "R1"
        self.commands.base_velocity.k_heading = 0.8
        self.commands.base_velocity.k_yaw_rate = 0.10
        self.commands.base_velocity.yaw_rate_limit = 0.30
        self.commands.base_velocity.low_pass_alpha = 0.15
        self.commands.base_velocity.yaw_rate_slew_limit = 0.01

        feet = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
        robot = SceneEntityCfg("robot")
        self.rewards.stand_horizontal_speed = RewTerm(
            func="g1_walk_centered.tasks.stage2r_mdp:stand_horizontal_speed_l2",
            weight=-3.0,
            params={"command_name": "base_velocity", "threshold": 0.05},
        )
        self.rewards.stand_yaw_rate = RewTerm(
            func="g1_walk_centered.tasks.stage2r_mdp:stand_yaw_rate_l2",
            weight=-0.25,
            params={"command_name": "base_velocity", "threshold": 0.05},
        )
        self.rewards.stand_flight = RewTerm(
            func="g1_walk_centered.tasks.stage2r_mdp:stand_flight_penalty",
            weight=-2.0,
            params={"command_name": "base_velocity", "sensor_cfg": feet, "threshold": 0.05},
        )
        self.rewards.stand_double_support = RewTerm(
            func="g1_walk_centered.tasks.stage2r_mdp:stand_double_support_reward",
            weight=0.5,
            params={"command_name": "base_velocity", "sensor_cfg": feet, "threshold": 0.05},
        )
        self.rewards.ankle_pitch_effort_hinge = RewTerm(
            func="g1_walk_centered.tasks.stage2r_mdp:AnklePitchEffortHinge",
            weight=-0.25,
            params={"asset_cfg": robot, "threshold": 0.95},
        )
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.dof_pos_limits.weight = -0.1
        self.rewards.feet_slide.weight = -0.25


@configclass
class UnifiedStandWalkR0EnvCfg(UnifiedStandWalkBaseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.stage2r_phase = "R0"


@configclass
class UnifiedStandWalkR1EnvCfg(UnifiedStandWalkBaseEnvCfg):
    pass


@configclass
class UnifiedStandWalkR2EnvCfg(UnifiedStandWalkBaseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.stage2r_phase = "R2"


@configclass
class UnifiedStandWalkR3EnvCfg(UnifiedStandWalkBaseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.stage2r_phase = "R3"


@configclass
class UnifiedStandWalkR4EnvCfg(UnifiedStandWalkBaseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.stage2r_phase = "R4"


@configclass
class UnifiedStandWalkEvalEnvCfg(UnifiedStandWalkR4EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
