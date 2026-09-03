"""Register the exp_011 Stage 2 task without modifying Isaac Lab core."""

from __future__ import annotations

import sys

import gymnasium as gym

from . import agents

TASK_ID = "Isaac-Exp011-Go2-Bidirectional-0To2-v0"

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:Exp011Go2BidirectionalEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Exp011Go2BidirectionalPPORunnerCfg",
        },
    )


def register_envs() -> list[str]:
    """External callback used by Isaac Lab's training launcher."""
    return [argument for argument in sys.argv[1:] if "=" in argument]
