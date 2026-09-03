"""Evaluation-only perturbations for the frozen Stage 9 policy."""

from __future__ import annotations

from typing import Any

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import HfRandomUniformTerrainCfg, TerrainGeneratorCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


BASE_ROBOT_STATIC_FRICTION = 0.8
BASE_ROBOT_DYNAMIC_FRICTION = 0.6
BASE_TERRAIN_STATIC_FRICTION = 1.0
BASE_TERRAIN_DYNAMIC_FRICTION = 1.0


def apply_robustness_config(
    env_cfg: Any,
    *,
    num_envs: int,
    friction_scale: float = 1.0,
    mass_scale: float = 1.0,
    com_shift_x_m: float = 0.0,
    stiffness_scale: float = 1.0,
    damping_scale: float = 1.0,
    small_rough_terrain: bool = False,
) -> dict[str, Any]:
    """Apply deterministic, relative perturbations to a resolved Stage 9 Eval config."""

    material_params = env_cfg.events.physics_material.params
    base_static = float(material_params["static_friction_range"][0])
    base_dynamic = float(material_params["dynamic_friction_range"][0])
    material_params["static_friction_range"] = (
        base_static * friction_scale,
        base_static * friction_scale,
    )
    material_params["dynamic_friction_range"] = (
        base_dynamic * friction_scale,
        base_dynamic * friction_scale,
    )

    if mass_scale != 1.0:
        env_cfg.events.robustness_body_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (mass_scale, mass_scale),
                "operation": "scale",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )

    if com_shift_x_m != 0.0:
        env_cfg.events.robustness_torso_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {
                    "x": (com_shift_x_m, com_shift_x_m),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                },
            },
        )

    if stiffness_scale != 1.0 or damping_scale != 1.0:
        params: dict[str, Any] = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "operation": "scale",
            "distribution": "uniform",
        }
        if stiffness_scale != 1.0:
            params["stiffness_distribution_params"] = (stiffness_scale, stiffness_scale)
        if damping_scale != 1.0:
            params["damping_distribution_params"] = (damping_scale, damping_scale)
        env_cfg.events.robustness_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params=params,
        )

    if small_rough_terrain:
        terrain_material = env_cfg.scene.terrain.physics_material
        env_cfg.scene.terrain.terrain_type = "generator"
        env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=42,
            curriculum=False,
            # Reset yaw is randomized in the baseline Eval task.  A square tile
            # leaves 120 m from its center in every heading for a 100 m episode.
            size=(240.0, 240.0),
            border_width=2.0,
            num_rows=1,
            num_cols=num_envs,
            horizontal_scale=1.0,
            vertical_scale=0.0025,
            slope_threshold=0.75,
            sub_terrains={
                "small_random_rough": HfRandomUniformTerrainCfg(
                    proportion=1.0,
                    noise_range=(-0.01, 0.01),
                    noise_step=0.005,
                    downsampled_scale=1.0,
                    border_width=1.0,
                )
            },
        )
        env_cfg.scene.terrain.max_init_terrain_level = 0
        env_cfg.scene.terrain.use_terrain_origins = True
        env_cfg.scene.terrain.physics_material = terrain_material

    return {
        "robot_static_friction": base_static * friction_scale,
        "robot_dynamic_friction": base_dynamic * friction_scale,
        "terrain_static_friction": float(env_cfg.scene.terrain.physics_material.static_friction),
        "terrain_dynamic_friction": float(env_cfg.scene.terrain.physics_material.dynamic_friction),
        "friction_combine_mode": env_cfg.scene.terrain.physics_material.friction_combine_mode,
        "mass_scale": mass_scale,
        "mass_recompute_inertia": True,
        "torso_com_shift_x_m": com_shift_x_m,
        "stiffness_scale": stiffness_scale,
        "damping_scale": damping_scale,
        "small_rough_terrain": small_rough_terrain,
        "rough_height_range_m": [-0.01, 0.01] if small_rough_terrain else [0.0, 0.0],
        "rough_noise_step_m": 0.005 if small_rough_terrain else 0.0,
        "rough_horizontal_scale_m": 1.0 if small_rough_terrain else 0.0,
    }
