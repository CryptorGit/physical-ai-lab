"""RSL-RL configuration for the command-conditioned G1 policy."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from g1_flat_run.tasks.agents.rsl_rl_ppo_cfg import G1FlatRunPPORunnerCfg


@configclass
class G1ResidualActorCfg(RslRlMLPModelCfg):
    """Experiment-local structured actor; the critic remains the stock RSL-RL MLP."""

    class_name: str = "g1_command_skills.models:G1CommandResidualActor"
    legacy_observation_dim: int = 123
    command_observation_dim: int = 29
    command_embedding_dim: int = 32
    state_embedding_dim: int = 64
    residual_hidden_dims: list[int] = [64, 64]
    residual_scale: float = 0.25
    trainable_skill_ids: list[int] = [0]
    corrective_command_embedding_dim: int = 16
    corrective_state_embedding_dim: int = 32
    corrective_hidden_dims: list[int] = [32, 32]
    stop_correction_action_indices: list[int] = [2, 7, 8, 3, 4, 19, 20, 0, 1]
    stop_correction_action_scales: list[float] = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.015, 0.015]
    crouch_action_indices: list[int] = [0, 1, 11, 12, 15, 16]
    crouch_action_scales: list[float] = [0.25, 0.25, 0.25, 0.25, 0.15, 0.15]
    train_stop_correction: bool = False
    crouch_controller: str = "scripted_shallow_v1"
    learned_crouch_residual_enabled: bool = False
    step_over_controller: str = "scripted_step_over_v0_guarded"
    learned_step_over_residual_enabled: bool = False
    freeze_base: bool = True


@configclass
class G1CommandSkillsPPORunnerCfg(G1FlatRunPPORunnerCfg):
    """Frozen Stage-4 base plus fully independent skill residual routes."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "physical_ai_g1_command_skills"
        self.save_interval = 50
        self.actor = G1ResidualActorCfg(
            hidden_dims=[256, 128, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
            trainable_skill_ids=[0],
        )


@configclass
class G1CommandRunPPORunnerCfg(G1CommandSkillsPPORunnerCfg):
    pass


@configclass
class G1CommandTurnPPORunnerCfg(G1CommandSkillsPPORunnerCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.actor.trainable_skill_ids = [2]


@configclass
class G1CommandTurnFullPPORunnerCfg(G1CommandTurnPPORunnerCfg):
    pass


@configclass
class G1CommandStopPPORunnerCfg(G1CommandSkillsPPORunnerCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        # model_31 is the frozen parent.  Only a zero-initialized, tightly
        # bounded correction route is optimized during the continuation pilot.
        self.actor.trainable_skill_ids = []
        self.actor.train_stop_correction = True
        self.algorithm.learning_rate = 2.5e-4
        self.algorithm.schedule = "fixed"
        self.save_interval = 4


@configclass
class G1CommandCrouchPPORunnerCfg(G1CommandSkillsPPORunnerCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.actor.trainable_skill_ids = [3]
        self.actor.train_stop_correction = False
        self.algorithm.learning_rate = 5.0e-4
        self.algorithm.schedule = "fixed"
        self.save_interval = 4


@configclass
class G1CommandSequencePPORunnerCfg(G1CommandSkillsPPORunnerCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        # Sequence training must not rewrite already accepted skill routes.
        # Only the critic and exploration std remain trainable here.
        self.actor.trainable_skill_ids = []
