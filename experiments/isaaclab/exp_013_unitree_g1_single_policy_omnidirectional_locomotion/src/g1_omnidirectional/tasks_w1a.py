"""Phase W1A training-only task registration."""

from __future__ import annotations

import gymnasium as gym
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString

from g1_single_policy.tasks import Exp012G1Stage2NEnvCfg, Exp012G1Stage2NRunnerCfg


@configclass
class Exp013W1AEnvCfg(Exp012G1Stage2NEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0
        command = self.commands.base_velocity
        command.class_type = ResolvableString(
            "g1_omnidirectional.w1a_command:W1AContinuousTranslationCommand"
        )
        command.resampling_time_range = (4.0, 4.0)
        command.heading_command = False
        command.rel_heading_envs = 0.0
        command.rel_standing_envs = 0.0
        command.ranges.lin_vel_x = (-1.2, 1.2)
        command.ranges.lin_vel_y = (-0.8, 0.8)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.ranges.heading = None


@configclass
class Exp013W1ARunnerCfg(Exp012G1Stage2NRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "exp_013_g1_omnidirectional"
        self.run_name = "phase_w1a_all_direction_translation_walk"
        self.num_steps_per_env = 24
        self.max_iterations = 200
        self.save_interval = 20
        self.actor.class_name = "g1_omnidirectional.w1a_models:W1AFrozenStdMLPModel"
        self.algorithm.class_name = "g1_omnidirectional.w1a_models:W1AFixedPPO"
        self.algorithm.learning_rate = 1.5e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.desired_kl = None


TASK_ID = "Isaac-Exp013-G1-W1A-TranslationWalk-v0"
if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:Exp013W1AEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}:Exp013W1ARunnerCfg",
        },
    )
