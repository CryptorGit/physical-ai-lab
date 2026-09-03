"""W1A2 task registration without modifying protected W1A."""
import gymnasium as gym
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString
from g1_omnidirectional.tasks_w1a import Exp013W1AEnvCfg,Exp013W1ARunnerCfg
@configclass
class Exp013W1A2EnvCfg(Exp013W1AEnvCfg):
 def __post_init__(self):
  super().__post_init__(); self.commands.base_velocity.class_type=ResolvableString("g1_omnidirectional.w1a2_command:W1A2SpeedEnvelopeCommand"); self.episode_length_s=12.
@configclass
class Exp013W1A2RunnerCfg(Exp013W1ARunnerCfg):
 seed=20272021; num_steps_per_env=24; max_iterations=160
gym.register(id="Isaac-Exp013-G1-W1A2-SpeedEnvelope-v0",entry_point="isaaclab.envs:ManagerBasedRLEnv",disable_env_checker=True,kwargs={"env_cfg_entry_point":f"{__name__}:Exp013W1A2EnvCfg","rsl_rl_cfg_entry_point":f"{__name__}:Exp013W1A2RunnerCfg"})
