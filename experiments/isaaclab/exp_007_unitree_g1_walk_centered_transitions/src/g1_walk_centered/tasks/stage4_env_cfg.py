"""Stage 4 WALK_TO_STAND transition training environment."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString

from .stage2wb_env_cfg import WalkStabilizationEnvCfg


@configclass
class WalkToStandTransitionEnvCfg(WalkStabilizationEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 16.0
        self.commands.base_velocity.class_type = ResolvableString(
            "g1_walk_centered.tasks.stage4_mdp:WalkToStandCommand"
        )
        self.commands.base_velocity.ramp_duration_s = 1.6
        self.commands.base_velocity.transition_timeout_s = 4.0
        self.actions.joint_pos.class_type = ResolvableString(
            "g1_walk_centered.tasks.stage4_mdp:RoutedWalkToStandJointPositionAction"
        )
        self.actions.joint_pos.stand_checkpoint_path = ""
        self.actions.joint_pos.walk_checkpoint_path = ""
        self.actions.joint_pos.stand_to_walk_checkpoint_path = ""
        self.rewards.completion_bonus = RewTerm(
            func="g1_walk_centered.tasks.stage4_mdp:completion_bonus",
            weight=12.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.source_alignment = RewTerm(
            func="g1_walk_centered.tasks.stage4_mdp:endpoint_alignment",
            weight=-0.10,
            params={"command_name": "base_velocity", "boundary": "source"},
        )
        self.rewards.target_alignment = RewTerm(
            func="g1_walk_centered.tasks.stage4_mdp:endpoint_alignment",
            weight=-0.10,
            params={"command_name": "base_velocity", "boundary": "target"},
        )
        self.rewards.reverse_motion = RewTerm(
            func="g1_walk_centered.tasks.stage4_mdp:reverse_motion",
            weight=-2.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.double_support_progress = RewTerm(
            func="g1_walk_centered.tasks.stage4_mdp:double_support_progress",
            weight=0.5,
            params={"command_name": "base_velocity"},
        )
        self.rewards.heading_error.func = (
            "g1_walk_centered.tasks.stage4_mdp:transition_masked_heading_error"
        )
        self.rewards.heading_error.params = {"command_name": "base_velocity"}
        self.rewards.lateral_velocity.func = (
            "g1_walk_centered.tasks.stage4_mdp:transition_lateral_velocity"
        )
        self.rewards.lateral_velocity.params = {"command_name": "base_velocity"}
        self.terminations.transition_completed = DoneTerm(
            func="g1_walk_centered.tasks.stage4_mdp:transition_completed",
            params={"command_name": "base_velocity"},
        )
        self.terminations.transition_failed = DoneTerm(
            func="g1_walk_centered.tasks.stage4_mdp:transition_failed",
            params={"command_name": "base_velocity"},
        )


@configclass
class WalkToStandTransitionEvalEnvCfg(WalkToStandTransitionEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
