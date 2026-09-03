"""W1A actor/PPO types with a frozen gait-conditioned Gaussian std."""

from __future__ import annotations

from g1_single_policy.stage2n_models import GaitAnchoredPPO, GaitConditionedMLPModel


class W1AFrozenStdMLPModel(GaitConditionedMLPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distribution.log_std_walk.requires_grad_(False)
        self.distribution.log_std_run.requires_grad_(False)


class W1AFixedPPO(GaitAnchoredPPO):
    """Standard PPO semantics with no anchor and a fixed LR."""

    anchor_beta = 0.0

