"""W1B yaw-conditioned omnidirectional WALK task registration."""
import gymnasium as gym
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString
from g1_omnidirectional.tasks_w1a2 import Exp013W1A2EnvCfg, Exp013W1A2RunnerCfg


@configclass
class Exp013W1BEnvCfg(Exp013W1A2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.class_type = ResolvableString(
            "g1_omnidirectional.w1b_command:W1BYawConditionedCommand"
        )
        self.commands.base_velocity.ranges.ang_vel_z = (-.5, .5)


@configclass
class Exp013W1BRunnerCfg(Exp013W1A2RunnerCfg):
    seed = 20274021
    num_steps_per_env = 24
    max_iterations = 200


gym.register(
    id="Isaac-Exp013-G1-W1B-YawWalk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Exp013W1BEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}:Exp013W1BRunnerCfg",
    },
)
