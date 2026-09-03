"""Import-light configuration for the Pathfinder standing task.

This module is loaded before SimulationApp starts, so it intentionally avoids
runtime simulation imports such as spawn helpers and math utilities.
"""
from __future__ import annotations

from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.configclass import configclass


REPO_ROOT = Path(__file__).resolve().parents[7]
USD_PATH = REPO_ROOT / "shared" / "models" / "pathfinder" / "usd" / "pathfinder_articulation.usd"

if not USD_PATH.is_file():
    raise FileNotFoundError(f"Pathfinder USD not found: {USD_PATH}")


@configclass
class PathfinderStandEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 5.0

    action_scale = 0.35
    action_space = 10
    observation_space = 39
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(usd_path=str(USD_PATH)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=150.0,
                velocity_limit_sim=8.0,
                stiffness=40.0,
                damping=4.0,
            )
        },
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=3.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    joint_noise = 0.015
    max_tilt_rad = 0.65
    min_height_ratio = 0.70

    rew_alive = 0.2
    rew_upright = 4.0
    rew_height = 2.0

    rew_joint_pose = -0.02
    rew_joint_vel = -0.002
    rew_action = -0.005
    rew_action_rate = -0.01

    rew_termination = -10.0
