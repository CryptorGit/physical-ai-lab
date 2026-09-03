"""EXP 012 task registration and frozen contracts."""

from __future__ import annotations

import sys
import gymnasium as gym
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString

from g1_flat_run.tasks.g1_flat_run_env_cfg import G1FlatRunStage2EnvCfg
from g1_flat_run.tasks.agents.rsl_rl_ppo_cfg import G1FlatRunPPORunnerCfg


@configclass
class Exp012G1EnvCfg(G1FlatRunStage2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Preserve every Stage 2 base term/weight. Add only the existing exp_005
        # Stage 4 safe periodic-flight implementation.
        self.rewards.safe_periodic_flight = RewTerm(
            func="g1_flat_run.tasks.stage3_mdp:SafePeriodicFlightReward",
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "min_command_speed": 2.3,
                "max_tracking_error": 0.30,
                "max_torso_tilt_rad": 0.20,
                "max_vertical_speed": 0.50,
                "min_flight_time": 0.04,
                "max_flight_time": 0.16,
                "precursor_reward_per_step": 0.25,
                "takeoff_precursor_reward_per_step": 0.05,
                "precursor_event_cap": 0.75,
                "precursor_min_flight_time": 0.04,
                "precursor_max_tracking_error": 1.20,
                "completion_reward": 2.0,
                "excess_flight_penalty_per_step": 0.25,
                "use_yaw_frame_tracking": True,
            },
        )
        command = self.commands.base_velocity
        command.class_type = ResolvableString(
            "g1_single_policy.command_curriculum:G1BidirectionalVelocityCommand")
        command.resampling_time_range = (1.0e9, 1.0e9)
        command.heading_command = False
        command.rel_heading_envs = 0.0
        command.rel_standing_envs = 0.0
        command.ranges.lin_vel_x = (0.0, 2.6)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.1, 0.1)
        command.ranges.heading = None


@configclass
class Exp012G1RunnerCfg(G1FlatRunPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "exp_012_g1_single_policy_bidirectional"
        self.run_name = "stage2_pilot1"
        self.save_interval = 25


@configclass
class Exp012G1PhaseAEnvCfg(Exp012G1EnvCfg):
    """Focused single-checkpoint continuation for RUN completion acquisition."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.commands.base_velocity.class_type = ResolvableString(
            "g1_single_policy.command_curriculum:G1PhaseARunAcquisitionCommand"
        )


@configclass
class Exp012G1PhaseARunnerCfg(Exp012G1RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.run_name = "stage2e_phase_a_run_acquisition_preflight"
        self.save_interval = 10


@configclass
class Exp012G1ReversePhaseR1EnvCfg(Exp012G1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.commands.base_velocity.class_type = ResolvableString(
            "g1_single_policy.command_curriculum:G1ReversePhaseR1Command"
        )


@configclass
class Exp012G1ReversePhaseR1RunnerCfg(Exp012G1RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.run_name = "stage2i_reverse_continuation_phase_r1"
        self.save_interval = 10


@configclass
class Exp012G1Stage2NEnvCfg(Exp012G1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0


@configclass
class Exp012G1Stage2NRunnerCfg(Exp012G1RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.run_name = "stage2n_gait_conditioned_ppo_retention_preflight"
        self.save_interval = 5
        self.actor.class_name = "g1_single_policy.stage2n_models:GaitConditionedMLPModel"
        self.actor.distribution_cfg.class_name = (
            "g1_single_policy.stage2n_models:GaitConditionedDiagonalGaussian"
        )
        self.algorithm.class_name = "g1_single_policy.stage2n_models:GaitAnchoredPPO"


TASK_ID = "Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0"
if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:Exp012G1EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}:Exp012G1RunnerCfg",
        },
    )


def register_envs():
    return [arg for arg in sys.argv[1:] if "=" in arg]


PHASE_A_TASK_ID = "Isaac-Exp012-G1-PhaseA-RunAcquisition-v0"
if PHASE_A_TASK_ID not in gym.registry:
    gym.register(
        id=PHASE_A_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:Exp012G1PhaseAEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}:Exp012G1PhaseARunnerCfg",
        },
    )


PHASE_R1_TASK_ID = "Isaac-Exp012-G1-Reverse-PhaseR1-v0"
if PHASE_R1_TASK_ID not in gym.registry:
    gym.register(
        id=PHASE_R1_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:Exp012G1ReversePhaseR1EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}:Exp012G1ReversePhaseR1RunnerCfg",
        },
    )


STAGE2N_TASK_ID = "Isaac-Exp012-G1-GaitPpoRetention-v0"
if STAGE2N_TASK_ID not in gym.registry:
    gym.register(
        id=STAGE2N_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:Exp012G1Stage2NEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}:Exp012G1Stage2NRunnerCfg",
        },
    )
