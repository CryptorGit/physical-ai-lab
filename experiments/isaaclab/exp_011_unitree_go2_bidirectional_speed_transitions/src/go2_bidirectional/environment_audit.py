"""Registry and live environment configuration audit."""

from __future__ import annotations

import gymnasium as gym


def registered_go2_specs() -> list[dict]:
    rows = []
    for spec in sorted(gym.registry.values(), key=lambda value: value.id):
        text = spec.id.lower()
        if "go2" not in text:
            continue
        rows.append(
            {
                "registered_environment_id": spec.id,
                "environment_config_class": spec.kwargs.get("env_cfg_entry_point"),
                "runner_config_class": spec.kwargs.get("rsl_rl_cfg_entry_point"),
                "entry_point": str(spec.entry_point),
            }
        )
    return rows


def live_environment_details(cfg) -> dict:
    return {
        "asset_config": type(cfg.scene.robot).__name__,
        "asset_usd": cfg.scene.robot.spawn.usd_path,
        "observation_manager": type(cfg.observations).__name__,
        "action_manager": type(cfg.actions).__name__,
        "reward_configuration": type(cfg.rewards).__name__,
        "termination_configuration": type(cfg.terminations).__name__,
        "command_configuration": type(cfg.commands).__name__,
        "training_command_range": {
            "lin_vel_x": list(cfg.commands.base_velocity.ranges.lin_vel_x),
            "lin_vel_y": list(cfg.commands.base_velocity.ranges.lin_vel_y),
            "ang_vel_z": list(cfg.commands.base_velocity.ranges.ang_vel_z),
            "heading": list(cfg.commands.base_velocity.ranges.heading),
        },
        "heading_command": cfg.commands.base_velocity.heading_command,
        "control_decimation": cfg.decimation,
        "physics_dt_s": cfg.sim.dt,
        "control_dt_s": cfg.sim.dt * cfg.decimation,
        "episode_duration_s": cfg.episode_length_s,
        "terrain_type": cfg.scene.terrain.terrain_type,
    }

