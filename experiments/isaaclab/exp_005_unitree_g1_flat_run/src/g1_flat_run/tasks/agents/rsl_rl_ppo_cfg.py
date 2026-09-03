"""RSL-RL configuration derived from the official G1 flat PPO runner."""

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatPPORunnerCfg,
)


@configclass
class G1FlatRunPPORunnerCfg(G1FlatPPORunnerCfg):
    """Keep the official network and PPO hyperparameters under a local log root."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "physical_ai_g1_flat_run"
