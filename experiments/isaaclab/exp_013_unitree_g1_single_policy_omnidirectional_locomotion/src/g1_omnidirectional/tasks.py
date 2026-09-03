"""EXP 013 evaluation-only task registration."""

from __future__ import annotations

import gymnasium as gym
from isaaclab.utils.configclass import configclass

from g1_single_policy.tasks import Exp012G1Stage2NEnvCfg, Exp012G1Stage2NRunnerCfg


@configclass
class Exp013DirectionalBaselineEnvCfg(Exp012G1Stage2NEnvCfg):
    """Expose the inherited velocity command to the complete Stage 0 domain."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        command = self.commands.base_velocity
        command.resampling_time_range = (1.0e9, 1.0e9)
        command.heading_command = False
        command.rel_heading_envs = 0.0
        command.rel_standing_envs = 0.0
        command.ranges.lin_vel_x = (-2.4, 2.4)
        command.ranges.lin_vel_y = (-2.4, 2.4)
        command.ranges.ang_vel_z = (-1.0, 1.0)
        command.ranges.heading = None


@configclass
class Exp013DirectionalBaselineRunnerCfg(Exp012G1Stage2NRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "exp_013_g1_omnidirectional"
        self.run_name = "stage0_parent_directional_baseline"


TASK_ID = "Isaac-Exp013-G1-DirectionalBaseline-v0"
if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:Exp013DirectionalBaselineEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}:Exp013DirectionalBaselineRunnerCfg",
        },
    )
