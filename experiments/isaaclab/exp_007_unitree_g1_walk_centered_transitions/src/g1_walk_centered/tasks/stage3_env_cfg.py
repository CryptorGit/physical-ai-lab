"""Stage 3 STAND_TO_WALK transition training environment."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString

from .stage2wb_env_cfg import WalkStabilizationEnvCfg


@configclass
class StandToWalkTransitionEnvCfg(WalkStabilizationEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 9.0
        self.commands.base_velocity.class_type = ResolvableString(
            "g1_walk_centered.tasks.stage3_mdp:StandToWalkCommand"
        )
        self.commands.base_velocity.ramp_duration_s = 1.5
        self.commands.base_velocity.transition_timeout_s = 4.0
        self.actions.joint_pos.class_type = ResolvableString(
            "g1_walk_centered.tasks.stage3_mdp:RoutedTransitionJointPositionAction"
        )
        self.actions.joint_pos.stand_checkpoint_path = ""
        self.actions.joint_pos.walk_checkpoint_path = ""
        self.rewards.completion_bonus = RewTerm(
            func="g1_walk_centered.tasks.stage3_mdp:completion_bonus",
            weight=10.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.source_alignment = RewTerm(
            func="g1_walk_centered.tasks.stage3_mdp:endpoint_alignment",
            weight=-0.10,
            params={"command_name": "base_velocity", "boundary": "source"},
        )
        self.rewards.target_alignment = RewTerm(
            func="g1_walk_centered.tasks.stage3_mdp:endpoint_alignment",
            weight=-0.10,
            params={"command_name": "base_velocity", "boundary": "target"},
        )
        self.rewards.heading_error.func = "g1_walk_centered.tasks.stage3_mdp:transition_masked_heading_error"
        self.rewards.heading_error.params = {"command_name": "base_velocity"}
        self.rewards.lateral_velocity.func = "g1_walk_centered.tasks.stage3_mdp:transition_lateral_velocity"
        self.rewards.lateral_velocity.params = {"command_name": "base_velocity"}
        self.terminations.transition_completed = DoneTerm(
            func="g1_walk_centered.tasks.stage3_mdp:transition_completed",
            params={"command_name": "base_velocity"},
        )
        self.terminations.transition_failed = DoneTerm(
            func="g1_walk_centered.tasks.stage3_mdp:transition_failed",
            params={"command_name": "base_velocity"},
        )


@configclass
class StandToWalkTransitionEvalEnvCfg(StandToWalkTransitionEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
