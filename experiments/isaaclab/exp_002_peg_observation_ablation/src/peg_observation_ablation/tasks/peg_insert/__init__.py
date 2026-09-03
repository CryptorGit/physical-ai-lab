"""Gym registrations for peg-insertion observation experiments."""

import gymnasium as gym


_ENV_ENTRY_POINT = (
    "peg_observation_ablation.tasks.peg_insert.peg_insert_env:"
    "PegObservationAblationEnv"
)

_CFG_MODULE = (
    "peg_observation_ablation.tasks.peg_insert.peg_insert_env_cfg:"
)

_RL_GAMES_CFG = (
    "isaaclab_tasks.direct.factory.agents:"
    "rl_games_ppo_cfg.yaml"
)


def register_task(task_id: str, config_class_name: str) -> None:
    """Register one observation-ablation task."""

    gym.register(
        id=task_id,
        entry_point=_ENV_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _CFG_MODULE + config_class_name,
            "rl_games_cfg_entry_point": _RL_GAMES_CFG,
        },
    )


register_task(
    "Isaac-PegObservationBaseline-Direct-v0",
    "PegObservationBaselineEnvCfg",
)

register_task(
    "Isaac-PegObservationNoAngvel-Direct-v0",
    "PegObservationNoAngvelEnvCfg",
)

register_task(
    "Isaac-PegObservationNoLinvel-Direct-v0",
    "PegObservationNoLinvelEnvCfg",
)

register_task(
    "Isaac-PegObservationNoVelocity-Direct-v0",
    "PegObservationNoVelocityEnvCfg",
)

register_task(
    "Isaac-PegObservationPositionOnly-Direct-v0",
    "PegObservationPositionOnlyEnvCfg",
)