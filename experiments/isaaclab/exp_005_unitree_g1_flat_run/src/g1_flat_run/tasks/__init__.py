"""Register the local Unitree G1 flat-running tasks."""

from __future__ import annotations

import sys

import gymnasium as gym

from . import agents


_TASKS = {
    "Isaac-Velocity-Flat-G1-Run-Stage1-v0": "G1FlatRunStage1EnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage2-v0": "G1FlatRunStage2EnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage3-v0": "G1FlatRunStage3EnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage3-Play-v0": "G1FlatRunStage3PlayEnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage3-Eval-v0": "G1FlatRunStage3EvalEnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage4-v0": "G1FlatRunStage4EnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage4-Play-v0": "G1FlatRunStage4PlayEnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage4-Eval-v0": "G1FlatRunStage4EvalEnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage5-v0": "G1FlatRunStage5EnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage5-Play-v0": "G1FlatRunStage5PlayEnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage5-Eval-v0": "G1FlatRunStage5EvalEnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage6-v0": "G1FlatRunStage6EnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage6-Play-v0": "G1FlatRunStage6PlayEnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage6-Eval-v0": "G1FlatRunStage6EvalEnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage7-v0": "G1FlatRunStage7EnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage7-Play-v0": "G1FlatRunStage7PlayEnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage7-Eval-v0": "G1FlatRunStage7EvalEnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage8-v0": "G1FlatRunStage8EnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage8-Play-v0": "G1FlatRunStage8PlayEnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage8-Eval-v0": "G1FlatRunStage8EvalEnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage9-v0": "G1FlatRunStage9EnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage9-Play-v0": "G1FlatRunStage9PlayEnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Stage9-Eval-v0": "G1FlatRunStage9EvalEnvCfg",
    "Isaac-Velocity-Flat-G1-Run-v0": "G1FlatRunEnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Play-v0": "G1FlatRunPlayEnvCfg",
    "Isaac-Velocity-Flat-G1-Run-Eval-v0": "G1FlatRunEvalEnvCfg",
}

for task_id, cfg_name in _TASKS.items():
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.g1_flat_run_env_cfg:{cfg_name}",
                "rsl_rl_cfg_entry_point": (
                    f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRunPPORunnerCfg"
                ),
            },
        )


def register_envs() -> list[str]:
    """Register tasks for Isaac Lab's external callback and preserve Hydra tokens."""
    return [argument for argument in sys.argv[1:] if "=" in argument]
