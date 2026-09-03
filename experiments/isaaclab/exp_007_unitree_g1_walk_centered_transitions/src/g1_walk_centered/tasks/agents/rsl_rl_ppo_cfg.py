"""PPO configuration preserving the parent Stage 2 architecture."""

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatPPORunnerCfg,
)


@configclass
class UnifiedStandWalkPPORunnerCfg(G1FlatPPORunnerCfg):
    """Frozen Stage 2R optimizer and network configuration."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "physical_ai_g1_walk_centered"
        self.save_interval = 50
        self.actor.distribution_cfg.init_std = 1.0
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.entropy_coef = 0.002


@configclass
class IndependentWalkPPORunnerCfg(UnifiedStandWalkPPORunnerCfg):
    """Stage 2W steady-WALK-only PPO configuration."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.run_name = "stage2w_independent_walk"
