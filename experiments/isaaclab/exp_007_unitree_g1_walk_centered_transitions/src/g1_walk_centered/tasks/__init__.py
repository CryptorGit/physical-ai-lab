"""Task registration and evaluation helpers for exp_007."""

from __future__ import annotations

import sys

import gymnasium as gym

from . import agents
from .evaluation import GATE_THRESHOLDS, classify_failures, retention_vs_exp006

REGISTERED_TASKS = (
    "Isaac-Velocity-Flat-G1-WalkCentered-R0-v0",
    "Isaac-Velocity-Flat-G1-WalkCentered-R1-v0",
    "Isaac-Velocity-Flat-G1-WalkCentered-R2-v0",
    "Isaac-Velocity-Flat-G1-WalkCentered-R3-v0",
    "Isaac-Velocity-Flat-G1-WalkCentered-R4-v0",
    "Isaac-Velocity-Flat-G1-WalkCentered-Eval-v0",
    "Isaac-Velocity-Flat-G1-IndependentWalk-v0",
    "Isaac-Velocity-Flat-G1-IndependentWalk-Eval-v0",
    "Isaac-Velocity-Flat-G1-WalkStabilization-v0",
    "Isaac-Velocity-Flat-G1-StandToWalk-v0",
    "Isaac-Velocity-Flat-G1-StandToWalk-Eval-v0",
    "Isaac-Velocity-Flat-G1-WalkToStand-v0",
    "Isaac-Velocity-Flat-G1-WalkToStand-Eval-v0",
)

_CONFIGS = {
    REGISTERED_TASKS[0]: "UnifiedStandWalkR0EnvCfg",
    REGISTERED_TASKS[1]: "UnifiedStandWalkR1EnvCfg",
    REGISTERED_TASKS[2]: "UnifiedStandWalkR2EnvCfg",
    REGISTERED_TASKS[3]: "UnifiedStandWalkR3EnvCfg",
    REGISTERED_TASKS[4]: "UnifiedStandWalkR4EnvCfg",
    REGISTERED_TASKS[5]: "UnifiedStandWalkEvalEnvCfg",
    REGISTERED_TASKS[6]: "IndependentWalkEnvCfg",
    REGISTERED_TASKS[7]: "IndependentWalkEvalEnvCfg",
    REGISTERED_TASKS[8]: "WalkStabilizationEnvCfg",
    REGISTERED_TASKS[9]: "StandToWalkTransitionEnvCfg",
    REGISTERED_TASKS[10]: "StandToWalkTransitionEvalEnvCfg",
    REGISTERED_TASKS[11]: "WalkToStandTransitionEnvCfg",
    REGISTERED_TASKS[12]: "WalkToStandTransitionEvalEnvCfg",
}

for task_id, cfg_name in _CONFIGS.items():
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    f"{__name__}.stage3_env_cfg:{cfg_name}"
                    if "StandToWalk" in cfg_name
                    else f"{__name__}.stage4_env_cfg:{cfg_name}"
                    if "WalkToStand" in cfg_name
                    else f"{__name__}.stage2wb_env_cfg:{cfg_name}"
                    if "Stabilization" in cfg_name
                    else f"{__name__}.stage2w_env_cfg:{cfg_name}"
                    if "IndependentWalk" in cfg_name
                    else f"{__name__}.stage2r_env_cfg:{cfg_name}"
                ),
                "rsl_rl_cfg_entry_point": (
                    f"{agents.__name__}.rsl_rl_ppo_cfg:"
                    + (
                        "IndependentWalkPPORunnerCfg"
                        if "IndependentWalk" in cfg_name or "Stabilization" in cfg_name or "StandToWalk" in cfg_name or "WalkToStand" in cfg_name
                        else "UnifiedStandWalkPPORunnerCfg"
                    )
                ),
            },
        )


def register_envs() -> list[str]:
    """Register external tasks while preserving Isaac Lab preset tokens."""
    return [argument for argument in sys.argv[1:] if "=" in argument]


__all__ = [
    "GATE_THRESHOLDS",
    "REGISTERED_TASKS",
    "classify_failures",
    "register_envs",
    "retention_vs_exp006",
]
