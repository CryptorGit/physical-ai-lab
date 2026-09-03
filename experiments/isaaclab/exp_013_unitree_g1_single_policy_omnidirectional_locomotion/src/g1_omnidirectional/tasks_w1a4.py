"""W1A4 task registration; protected W1A/W1A2 configs remain untouched."""
import gymnasium as gym
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString
from g1_omnidirectional.tasks_w1a2 import Exp013W1A2EnvCfg, Exp013W1A2RunnerCfg


@configclass
class Exp013W1A4EnvCfg(Exp013W1A2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.class_type = ResolvableString(
            "g1_omnidirectional.w1a4_command:W1A4RetentionCommand")


@configclass
class Exp013W1A4RunnerCfg(Exp013W1A2RunnerCfg):
    seed = 20273021
    num_steps_per_env = 24
    max_iterations = 60


gym.register(id="Isaac-Exp013-G1-W1A4-Retention-v0",
             entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": f"{__name__}:Exp013W1A4EnvCfg",
                     "rsl_rl_cfg_entry_point": f"{__name__}:Exp013W1A4RunnerCfg"})
