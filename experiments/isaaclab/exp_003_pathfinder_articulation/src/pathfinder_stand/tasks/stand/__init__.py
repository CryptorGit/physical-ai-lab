"""Pathfinder standing task registration."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Pathfinder-Stand-Direct-v0",
    entry_point=f"{__name__}.pathfinder_stand_env:PathfinderStandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pathfinder_stand_env_cfg:PathfinderStandEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
