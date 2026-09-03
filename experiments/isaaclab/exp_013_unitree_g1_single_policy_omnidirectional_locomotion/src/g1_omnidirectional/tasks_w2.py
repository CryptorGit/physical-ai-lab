"""Phase W2 dynamic omnidirectional WALK transition task."""
import gymnasium as gym

from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString

from g1_omnidirectional.tasks_w1b_r2 import Exp013W1BR2EnvCfg, Exp013W1BR2RunnerCfg
from g1_omnidirectional import w2_mdp


@configclass
class Exp013W2EnvCfg(Exp013W1BR2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.class_type = ResolvableString(
            "g1_omnidirectional.w2_command:W2DynamicSequenceCommand"
        )
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.episode_length_s = 45.0
        self.observations.policy.velocity_commands.func = w2_mdp.actor_velocity_command
        self.rewards.track_lin_vel_xy_exp.func = w2_mdp.track_lin_vel_xy_physical
        self.rewards.track_ang_vel_z_exp.func = w2_mdp.track_ang_vel_z_physical


@configclass
class Exp013W2RunnerCfg(Exp013W1BR2RunnerCfg):
    seed = 20275021
    num_steps_per_env = 24
    max_iterations = 250


gym.register(
    id="Isaac-Exp013-G1-W2-DynamicWalk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Exp013W2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}:Exp013W2RunnerCfg",
    },
)
