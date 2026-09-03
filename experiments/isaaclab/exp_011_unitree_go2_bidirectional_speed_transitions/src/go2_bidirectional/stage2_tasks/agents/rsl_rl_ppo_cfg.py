"""Official Go2 flat runner with logging/checkpoint cadence changes only."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)


@configclass
class Exp011Go2BidirectionalPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "exp_011_go2_continuous_0_to_2"
        self.run_name = "pilot1_seed20260911"
        self.seed = 20260911
        self.max_iterations = 300
        self.save_interval = 25
