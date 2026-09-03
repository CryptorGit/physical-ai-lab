"""Configurations derived from Isaac Lab's official Unitree G1 flat task."""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg


@configclass
class G1FlatRunBaseEnvCfg(G1FlatEnvCfg):
    """Shared straight-running changes while preserving observation and action spaces."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Prefer forward motion with only small lateral and yaw commands.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)

        # Keep the official reward terms. Only tune weights needed for the
        # walking-to-running transition; strict flight phase remains evaluation-only.
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.params["threshold"] = 0.25
        self.rewards.feet_slide.weight = -0.2


@configclass
class G1FlatRunStage1EnvCfg(G1FlatRunBaseEnvCfg):
    """Stage 1: extend the official 0-1 m/s walking range to 0-1.5 m/s."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.5)


@configclass
class G1FlatRunStage2EnvCfg(G1FlatRunBaseEnvCfg):
    """Stage 2: expose the policy to transitional speeds up to 2.2 m/s."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.2)


@configclass
class G1FlatRunStage3EnvCfg(G1FlatRunBaseEnvCfg):
    """Stage 3: concentrate on safe periodic running around 2.4--2.5 m/s."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Resolve the custom term only after SimulationApp starts. Importing the
        # command implementation while task configs are discovered preloads PXR.
        self.commands.base_velocity.class_type = ResolvableString(
            "g1_flat_run.tasks.stage3_mdp:FocusedRunVelocityCommand"
        )
        self.commands.base_velocity.ranges.lin_vel_x = (2.3, 2.6)

        # Increase slip cost conservatively from -0.20, and reward only a safe
        # alternating landing after a short flight (not raw airborne time).
        self.rewards.feet_slide.weight = -0.25
        self.rewards.safe_periodic_flight = RewTerm(
            func="g1_flat_run.tasks.stage3_mdp:SafePeriodicFlightReward",
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "min_command_speed": 2.3,
                "max_tracking_error": 0.30,
                "max_torso_tilt_rad": 0.20,
                "max_vertical_speed": 0.50,
                "min_flight_time": 0.04,
                "max_flight_time": 0.16,
            },
        )


@configclass
class G1FlatRunEnvCfg(G1FlatRunBaseEnvCfg):
    """Final stage: cover standing, walking, and high-speed commands up to 3 m/s."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 3.0)


@configclass
class G1FlatRunPlayEnvCfg(G1FlatRunEnvCfg):
    """Deterministic play configuration for the final-stage task."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class G1FlatRunEvalEnvCfg(G1FlatRunPlayEnvCfg):
    """Evaluation configuration; the evaluator writes fixed commands directly."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class G1FlatRunStage3PlayEnvCfg(G1FlatRunStage3EnvCfg):
    """Deterministic 2.5 m/s play configuration for Stage 3."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (2.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class G1FlatRunStage3EvalEnvCfg(G1FlatRunStage3PlayEnvCfg):
    """Stage 3 reward configuration with evaluator-controlled fixed commands."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class G1FlatRunStage4EnvCfg(G1FlatRunStage3EnvCfg):
    """Stage 4: dense-but-capped safe-flight shaping plus alternating landing completion."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.safe_periodic_flight.params.update(
            {
                "precursor_reward_per_step": 0.25,
                "takeoff_precursor_reward_per_step": 0.05,
                "precursor_event_cap": 0.75,
                "precursor_min_flight_time": 0.04,
                "precursor_max_tracking_error": 1.20,
                "completion_reward": 2.0,
                "excess_flight_penalty_per_step": 0.25,
                "use_yaw_frame_tracking": True,
            }
        )
@configclass
class G1FlatRunStage4PlayEnvCfg(G1FlatRunStage4EnvCfg):
    """Deterministic 2.5 m/s play configuration for Stage 4."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (2.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class G1FlatRunStage4EvalEnvCfg(G1FlatRunStage4PlayEnvCfg):
    """Stage 4 reward configuration with evaluator-controlled fixed commands."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class G1FlatRunStage5EnvCfg(G1FlatRunStage4EnvCfg):
    """Stage 5: success-gated expansion from 3.8 to 4.0 m/s."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.class_type = ResolvableString(
            "g1_flat_run.tasks.stage5_mdp:Stage5ProgressiveVelocityCommand"
        )
        self.commands.base_velocity.ranges.lin_vel_x = (3.4, 3.8)
        self.curriculum.speed_ceiling = CurrTerm(
            func="g1_flat_run.tasks.stage5_mdp:ProgressiveSpeedCurriculum",
            params={
                "command_name": "base_velocity",
                "safe_reward_name": "safe_periodic_flight",
                "diagnostic_reward_name": "high_speed_diagnostics",
                "stage_upper_bounds": (3.8, 3.9, 4.0),
                "success_threshold": 0.80,
                "min_samples": 50,
                "window_size": 100,
                "target_exposure_fraction": 0.50,
                "min_episode_fraction": 0.90,
                "min_safe_landings": 3,
                "max_slip_mps": 0.65,
                "max_velocity_limit_fraction": 0.05,
                "max_torque_limit_fraction": 0.20,
                "max_landing_impact_n": 3500.0,
                "max_vertical_excursion_m": 0.30,
                "max_asymmetry": 0.20,
                "initial_stage": 0,
            },
        )
        # The term returns exactly zero.  Weight 1 keeps it active so it can
        # collect diagnostics without changing the objective.
        self.rewards.high_speed_diagnostics = RewTerm(
            func="g1_flat_run.tasks.stage5_mdp:HighSpeedRunningDiagnostics",
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "saturation_ratio": 0.95,
            },
        )


@configclass
class G1FlatRunStage5PlayEnvCfg(G1FlatRunStage5EnvCfg):
    """Deterministic 4.0 m/s play configuration for Stage 5."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Evaluation/play commands are fixed explicitly and must not be rewritten
        # by the training-only speed curriculum during environment construction.
        self.curriculum.speed_ceiling = None
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (4.0, 4.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class G1FlatRunStage5EvalEnvCfg(G1FlatRunStage5PlayEnvCfg):
    """Stage 5 diagnostics with evaluator-controlled fixed commands."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class G1FlatRunStage6EnvCfg(G1FlatRunStage5EnvCfg):
    """Stage 6: reduce landing tails and actuator saturation at 3.8--4.0 m/s."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.curriculum.speed_ceiling = None
        self.commands.base_velocity.class_type = ResolvableString(
            "g1_flat_run.tasks.stage6_mdp:Stage6VelocityCommand"
        )
        self.commands.base_velocity.ranges.lin_vel_x = (3.8, 4.0)
        feet = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
        robot = SceneEntityCfg("robot")
        self.rewards.landing_impact = RewTerm(
            func="g1_flat_run.tasks.stage6_mdp:RobustLandingPenalty",
            weight=-0.25,
            params={
                "asset_cfg": robot,
                "sensor_cfg": feet,
                "force_threshold_n": 1000.0,
                "force_scale_n": 1000.0,
                "impact_component": 1.0,
                "velocity_component": 0.0,
            },
        )
        self.rewards.precontact_foot_velocity = RewTerm(
            func="g1_flat_run.tasks.stage6_mdp:RobustLandingPenalty",
            weight=-0.50,
            params={
                "asset_cfg": robot,
                "sensor_cfg": feet,
                "downward_speed_threshold_mps": 3.0,
                "downward_speed_scale_mps": 0.5,
                "impact_component": 0.0,
                "velocity_component": 1.0,
            },
        )
        self.rewards.joint_velocity_saturation = RewTerm(
            func="g1_flat_run.tasks.stage6_mdp:JointSaturationPenalty",
            weight=-0.10,
            params={"asset_cfg": robot, "quantity": "velocity", "threshold": 0.95, "scale": 0.05},
        )
        self.rewards.joint_torque_saturation = RewTerm(
            func="g1_flat_run.tasks.stage6_mdp:JointSaturationPenalty",
            weight=-0.10,
            params={"asset_cfg": robot, "quantity": "torque", "threshold": 0.95, "scale": 0.05},
        )
        self.rewards.landing_impact_symmetry = RewTerm(
            func="g1_flat_run.tasks.stage6_mdp:LandingImpactSymmetryPenalty",
            weight=-0.05,
            params={"sensor_cfg": feet, "normalization_n": 3500.0},
        )


@configclass
class G1FlatRunStage6PlayEnvCfg(G1FlatRunStage6EnvCfg):
    """Deterministic 4.0 m/s play configuration for Stage 6."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (4.0, 4.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class G1FlatRunStage6EvalEnvCfg(G1FlatRunStage6PlayEnvCfg):
    """Stage 6 reward configuration with evaluator-controlled fixed commands."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class G1FlatRunStage7EnvCfg(G1FlatRunStage6EnvCfg):
    """Stage 7: high-quality periodic running through 4.50 m/s."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.class_type = ResolvableString(
            "g1_flat_run.tasks.stage7_mdp:Stage7VelocityCommand"
        )
        self.commands.base_velocity.ranges.lin_vel_x = (4.25, 4.55)
        feet_sensor = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
        feet_asset = SceneEntityCfg("robot", body_names=".*_ankle_roll_link")
        self.rewards.high_speed_feet_slide = RewTerm(
            func="g1_flat_run.tasks.stage7_mdp:high_speed_feet_slide",
            weight=-0.05,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": feet_sensor,
                "asset_cfg": feet_asset,
                "start_speed_mps": 4.40,
                "full_speed_mps": 4.50,
            },
        )


@configclass
class G1FlatRunStage7PlayEnvCfg(G1FlatRunStage7EnvCfg):
    """Deterministic 4.50 m/s play configuration for Stage 7."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (4.50, 4.50)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class G1FlatRunStage7EvalEnvCfg(G1FlatRunStage7PlayEnvCfg):
    """Stage 7 reward configuration with evaluator-controlled fixed commands."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class G1FlatRunStage8EnvCfg(G1FlatRunStage6EnvCfg):
    """Stage 8: reduce high-speed excess slip without retraining from Stage 7."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.class_type = ResolvableString(
            "g1_flat_run.tasks.stage8_mdp:Stage8VelocityCommand"
        )
        self.commands.base_velocity.ranges.lin_vel_x = (4.30, 4.50)
        feet_sensor = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
        feet_asset = SceneEntityCfg("robot", body_names=".*_ankle_roll_link")
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func="g1_flat_run.tasks.stage8_mdp:quality_saturated_track_lin_vel_xy_yaw_frame_exp",
            weight=2.0,
            params={
                "command_name": "base_velocity",
                "std": 0.5,
                "acceptable_error_mps": 0.15,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.quality_gated_excess_slip = RewTerm(
            func="g1_flat_run.tasks.stage8_mdp:quality_gated_excess_slip",
            weight=-0.20,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": feet_sensor,
                "asset_cfg": feet_asset,
                "min_command_speed_mps": 4.40,
                "max_tracking_error_mps": 0.25,
                "slip_threshold_mps": 0.50,
            },
        )


@configclass
class G1FlatRunStage8PlayEnvCfg(G1FlatRunStage8EnvCfg):
    """Deterministic 4.50 m/s play configuration for Stage 8."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (4.50, 4.50)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class G1FlatRunStage8EvalEnvCfg(G1FlatRunStage8PlayEnvCfg):
    """Stage 8 reward configuration with evaluator-controlled commands."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class G1FlatRunStage9EnvCfg(G1FlatRunStage8EnvCfg):
    """Stage 9: stabilize slip and safe-cycle continuity around 5.0 m/s."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.class_type = ResolvableString(
            "g1_flat_run.tasks.stage9_mdp:Stage9VelocityCommand"
        )
        self.commands.base_velocity.ranges.lin_vel_x = (4.70, 5.10)
        self.rewards.safe_periodic_flight.params.update(
            {
                "continuous_tracking_decay": True,
                "tracking_forward_scale_mps": 0.30,
                "tracking_lateral_scale_mps": 0.20,
            }
        )
        self.rewards.knee_velocity_saturation = RewTerm(
            func="g1_flat_run.tasks.stage6_mdp:JointSaturationPenalty",
            weight=-0.10,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*_knee_joint"),
                "quantity": "velocity",
                "threshold": 0.95,
                "scale": 0.05,
            },
        )


@configclass
class G1FlatRunStage9PlayEnvCfg(G1FlatRunStage9EnvCfg):
    """Deterministic 5.0 m/s play configuration for Stage 9."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (5.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class G1FlatRunStage9EvalEnvCfg(G1FlatRunStage9PlayEnvCfg):
    """Stage 9 reward configuration with evaluator-controlled fixed commands."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
