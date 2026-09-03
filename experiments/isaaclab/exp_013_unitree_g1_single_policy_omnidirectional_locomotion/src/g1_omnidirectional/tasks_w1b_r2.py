"""W1B-R2 task registration with the repaired command sampler only."""
import gymnasium as gym

from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString

from g1_omnidirectional.tasks_w1b import Exp013W1BEnvCfg, Exp013W1BRunnerCfg


@configclass
class Exp013W1BR2EnvCfg(Exp013W1BEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.class_type = ResolvableString(
            "g1_omnidirectional.w1b_r2_command:W1BR2PendingMirrorCommand"
        )


@configclass
class Exp013W1BR2RunnerCfg(Exp013W1BRunnerCfg):
    pass


gym.register(
    id="Isaac-Exp013-G1-W1B-R2-YawWalk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Exp013W1BR2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}:Exp013W1BR2RunnerCfg",
    },
)
