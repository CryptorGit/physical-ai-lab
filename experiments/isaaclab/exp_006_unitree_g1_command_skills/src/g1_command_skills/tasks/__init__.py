"""Register Unitree G1 command-skill environments."""

from __future__ import annotations

import sys

import gymnasium as gym

from . import agents


_MODES = {
    "Run": ("G1CommandRunEnvCfg", "G1CommandRunPPORunnerCfg"),
    "Turn": ("G1CommandTurnEnvCfg", "G1CommandTurnPPORunnerCfg"),
    "TurnFull": ("G1CommandTurnFullEnvCfg", "G1CommandTurnFullPPORunnerCfg"),
    "Stop": ("G1CommandStopEnvCfg", "G1CommandStopPPORunnerCfg"),
    "StopB": ("G1CommandStopBEnvCfg", "G1CommandStopPPORunnerCfg"),
    "StopC": ("G1CommandStopCEnvCfg", "G1CommandStopPPORunnerCfg"),
    "Crouch": ("G1CommandCrouchEnvCfg", "G1CommandCrouchPPORunnerCfg"),
    "CrouchShallow": ("G1CommandCrouchShallowEnvCfg", "G1CommandCrouchPPORunnerCfg"),
    "StepOverAudit": ("G1CommandStepOverAuditEnvCfg", "G1CommandCrouchPPORunnerCfg"),
    "Sequence": ("G1CommandSequenceEnvCfg", "G1CommandSequencePPORunnerCfg"),
}

for label, (cfg_name, runner_cfg_name) in _MODES.items():
    for suffix, cfg_suffix in (("", ""), ("-Play", "Play"), ("-Eval", "Eval")):
        task_id = f"Isaac-Motion-Flat-G1-Command-{label}{suffix}-v0"
        resolved_cfg = cfg_name.replace("EnvCfg", f"{cfg_suffix}EnvCfg") if cfg_suffix else cfg_name
        if task_id not in gym.registry:
            gym.register(
                id=task_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": f"{__name__}.g1_command_env_cfg:{resolved_cfg}",
                    "rsl_rl_cfg_entry_point": (
                        f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_cfg_name}"
                    ),
                },
            )


def register_envs() -> list[str]:
    """External callback used by Isaac Lab while preserving Hydra overrides."""
    return [argument for argument in sys.argv[1:] if "=" in argument]
