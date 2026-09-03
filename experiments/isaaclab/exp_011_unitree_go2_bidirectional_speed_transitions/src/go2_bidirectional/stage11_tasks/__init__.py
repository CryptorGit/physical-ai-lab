"""Register the isolated Stage 11 tangential-slip reward task."""

from __future__ import annotations

import gymnasium as gym

from go2_bidirectional.stage2_tasks import agents

TASK_ID = "Isaac-Exp011-Go2-Tangential-Slip-v0"

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:Exp011Go2TangentialSlipEnvCfg",
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:Exp011Go2BidirectionalPPORunnerCfg"
            ),
        },
    )
