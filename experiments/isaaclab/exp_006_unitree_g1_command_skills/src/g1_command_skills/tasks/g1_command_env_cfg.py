"""Isaac Lab configurations for staged Unitree G1 command skills."""

from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import schemas as sim_schemas
from isaaclab.sim import spawners as sim_utils
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString

from g1_flat_run.tasks.g1_flat_run_env_cfg import G1FlatRunStage4EnvCfg


@configclass
class G1CommandBaseEnvCfg(G1FlatRunStage4EnvCfg):
    """Shared warm-start-compatible policy with command-gated skill rewards."""

    command_mode: str = "sequence"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 12.0

        # The first command observation remains exactly the legacy body-frame
        # [vx, vy, yaw-rate].  All new fields are appended after the old 123-D vector.
        command = self.commands.base_velocity
        command.class_type = ResolvableString("g1_command_skills.tasks.command_mdp:MotionCommand")
        command.mode = self.command_mode
        command.rehearsal_probabilities = {
            "run": (1.0, 0.0, 0.0, 0.0, 0.0),
            "turn": (0.30, 0.70, 0.0, 0.0, 0.0),
            "stop": (0.20, 0.20, 0.60, 0.0, 0.0),
            "sequence": (0.10, 0.10, 0.10, 0.70, 0.0),
            "crouch": (0.0, 0.0, 0.0, 0.0, 1.0),
        }[self.command_mode]
        command.run_speed_range = (1.2, 2.6)
        command.turn_speed_range = (1.2, 2.3)
        command.stop_distance_range = (0.5, 1.5)
        command.stop_entry_speed_floor_mps = 0.0
        command.stop_deceleration_range_mps2 = (0.20, 2.00)
        command.stop_target_radius_m = 0.15
        command.stop_hold_position_tolerance_m = 0.50
        command.stop_hold_speed_tolerance_mps = 0.20
        command.stop_hold_duration_s = 1.0
        command.stop_heading_dead_zone_rad = 0.02
        command.stop_heading_feedback_gain = 1.5
        command.stop_heading_yaw_rate_limit_rps = 0.45
        command.turn_angles_deg = (45.0, 90.0)
        command.turn_angle_probabilities = (0.5, 0.5)
        command.turn_direction_probabilities = (0.5, 0.5)
        command.deterministic_turn_evaluation = False
        command.transition_duration_s = 0.4
        command.single_skill_duration_s = 8.0
        command.turn_script_durations_s = (1.5, 3.5, 3.0)
        command.stop_script_durations_s = (2.0, 6.0)
        command.sequence_durations_s = (3.0, 2.5, 2.0, 3.5)
        command.phase_duration_jitter_fraction = 0.25
        command.path_lookahead_distance_m = 1.0
        command.run_path_initial_lateral_error_range_m = (-0.40, 0.40)
        command.standing_pelvis_height_m = 0.78
        command.crouch_height_drop_range_m = (0.08, 0.15)
        command.crouch_supported_depth_min_m = 0.08
        command.crouch_supported_depth_max_m = 0.10
        command.crouch_unsupported_command_mode = "reject"
        command.crouch_evaluation_depths_m = ()
        command.crouch_down_time_range_s = (1.0, 2.0)
        command.crouch_hold_time_range_s = (0.8, 1.2)
        command.crouch_return_time_range_s = (1.0, 2.0)
        command.crouch_stand_hold_time_range_s = (0.8, 1.2)
        command.crouch_nominal_duration_s = 5.0
        command.crouch_settle_timeout_s = 2.0
        command.crouch_settle_hold_s = 0.4
        command.crouch_settle_horizontal_speed_mps = 0.08
        command.crouch_settle_vertical_speed_mps = 0.05
        command.crouch_settle_tilt_rad = 0.10
        command.crouch_base_crossfade_duration_s = 0.4
        # Isolated Stage A starts directly on the standing option. Integrated
        # sequences can set this false and wait for the safety condition.
        command.crouch_standalone_standing_base = self.command_mode == "crouch"
        command.resampling_time_range = (1.0e9, 1.0e9)
        command.rel_standing_envs = 0.0
        command.rel_heading_envs = 1.0
        command.heading_command = True
        command.heading_control_stiffness = 1.5
        command.debug_vis = True
        command.ranges.lin_vel_x = (0.0, 2.6)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-1.5, 1.5)
        command.ranges.heading = (-3.141592653589793, 3.141592653589793)
        self.observations.policy.motion_command = ObsTerm(
            func="g1_command_skills.tasks.command_mdp:motion_command_observation",
            params={"command_name": "base_velocity"},
            clip=(-10.0, 10.0),
        )

        robot = SceneEntityCfg("robot")
        self.rewards.run_command_tracking = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:run_tracking",
            weight=2.0,
            params={"command_name": "base_velocity", "speed_std": 0.35, "heading_std": 0.25, "asset_cfg": robot},
        )
        self.rewards.run_path_lateral_error = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:run_path_lateral_error",
            weight=-1.0,
            params={"command_name": "base_velocity", "dead_zone_m": 0.20, "error_scale_m": 0.50},
        )
        self.rewards.run_path_lateral_velocity = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:run_path_lateral_velocity",
            weight=-0.20,
            params={"command_name": "base_velocity", "velocity_scale_mps": 0.50},
        )
        self.rewards.run_path_recovery_progress = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:RunPathRecoveryProgressReward",
            weight=0.50,
            params={
                "command_name": "base_velocity",
                "dead_zone_m": 0.20,
                "progress_scale_m_per_step": 0.05,
            },
        )
        self.rewards.turn_command_tracking = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:turn_tracking",
            weight=2.5,
            params={"command_name": "base_velocity", "speed_std": 0.40, "heading_std": 0.18, "asset_cfg": robot},
        )
        self.rewards.turn_side_slip = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:turn_side_slip",
            weight=-0.25,
            params={"command_name": "base_velocity", "asset_cfg": robot},
        )
        self.rewards.stop_position_velocity = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:stop_position_velocity",
            weight=3.0,
            params={
                "command_name": "base_velocity",
                "position_std": 0.30,
                "speed_std": 0.20,
                "asset_cfg": robot,
            },
        )
        self.rewards.stop_progress = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:StopProgressReward",
            weight=1.5,
            params={"command_name": "base_velocity", "progress_scale_m_per_step": 0.05},
        )
        self.rewards.stop_braking_speed_tracking = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:stop_braking_speed_tracking",
            weight=2.5,
            params={"command_name": "base_velocity", "speed_std": 0.25, "asset_cfg": robot},
        )
        self.rewards.stop_overshoot = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:stop_overshoot",
            weight=-2.0,
            params={"command_name": "base_velocity", "scale_m": 0.30},
        )
        self.rewards.stop_early = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:stop_early",
            weight=-0.75,
            params={"command_name": "base_velocity", "speed_margin_mps": 0.20, "scale_mps": 0.40, "asset_cfg": robot},
        )
        self.rewards.stop_heading = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:stop_heading",
            weight=1.5,
            params={"command_name": "base_velocity", "heading_std": 0.12},
        )
        self.rewards.stop_heading_tail = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:stop_heading_tail",
            weight=-0.50,
            params={"command_name": "base_velocity", "dead_zone_rad": 0.10, "scale_rad": 0.12},
        )
        self.rewards.stop_yaw_rate_tracking = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:stop_yaw_rate_tracking",
            weight=-0.25,
            params={"command_name": "base_velocity", "rate_scale_rps": 0.60, "asset_cfg": robot},
        )
        self.rewards.stop_attitude_stability = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:stop_attitude_stability",
            weight=-0.15,
            params={
                "command_name": "base_velocity",
                "tilt_scale": 0.20,
                "angular_rate_scale_rps": 1.0,
                "asset_cfg": robot,
            },
        )
        self.rewards.stop_instability_tail = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:stop_instability_tail",
            weight=-0.35,
            params={
                "command_name": "base_velocity",
                "tilt_threshold": 0.20,
                "yaw_rate_threshold_rps": 1.0,
                "scale": 0.20,
                "asset_cfg": robot,
            },
        )
        self.rewards.stop_hold_heading = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:stop_hold_heading",
            weight=0.75,
            params={"command_name": "base_velocity", "heading_std": 0.10},
        )
        self.rewards.stop_parent_action_deviation = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:stop_parent_action_deviation",
            weight=-2.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.stop_hold = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:stop_hold",
            weight=2.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.stop_joint_velocity_saturation = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:stop_joint_saturation",
            weight=-0.10,
            params={"command_name": "base_velocity", "asset_cfg": robot, "quantity": "velocity", "threshold": 0.95, "scale": 0.05},
        )
        self.rewards.stop_ankle_torque_saturation = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:stop_joint_saturation",
            weight=-0.15,
            params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle_.*_joint"), "quantity": "torque", "threshold": 0.95, "scale": 0.05},
        )
        self.rewards.stop_slip = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:stop_foot_slip",
            weight=-0.25,
            params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"), "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )
        self.rewards.stop_impact = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:stop_impact",
            weight=-0.10,
            params={"command_name": "base_velocity", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"), "force_threshold_n": 1000.0, "force_scale_n": 1000.0},
        )
        self.rewards.stop_upright = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:skill_upright",
            weight=1.0,
            params={"command_name": "base_velocity", "skill_ids": (1,), "tilt_std": 0.15, "asset_cfg": robot},
        )

        crouch_sagittal = SceneEntityCfg(
            "robot",
            joint_names=[
                "left_hip_pitch_joint", "right_hip_pitch_joint",
                "left_knee_joint", "right_knee_joint",
                "left_ankle_pitch_joint", "right_ankle_pitch_joint",
            ],
            preserve_order=True,
        )
        crouch_feet = SceneEntityCfg("robot", body_names=".*_ankle_roll_link")
        crouch_contacts = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
        self.rewards.crouch_height_tracking = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:crouch_height_tracking", weight=4.0,
            params={"command_name": "base_velocity", "height_std_m": 0.035, "asset_cfg": robot},
        )
        self.rewards.crouch_vertical_velocity = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:crouch_vertical_velocity_tracking", weight=1.0,
            params={"command_name": "base_velocity", "velocity_std_mps": 0.12, "asset_cfg": robot},
        )
        self.rewards.crouch_depth_progress = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:CrouchDepthProgressReward", weight=1.0,
            params={"command_name": "base_velocity", "progress_scale_m_per_step": 0.01},
        )
        self.rewards.crouch_hold_height = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:crouch_hold_height", weight=2.0,
            params={"command_name": "base_velocity", "height_std_m": 0.03, "asset_cfg": robot},
        )
        self.rewards.crouch_return_height = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:crouch_return_height", weight=2.5,
            params={"command_name": "base_velocity", "height_std_m": 0.04, "asset_cfg": robot},
        )
        self.rewards.crouch_upright = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:skill_upright", weight=1.5,
            params={"command_name": "base_velocity", "skill_ids": (3,), "tilt_std": 0.12, "asset_cfg": robot},
        )
        self.rewards.crouch_symmetry = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:crouch_joint_symmetry", weight=-0.20,
            params={"command_name": "base_velocity", "asset_cfg": crouch_sagittal, "difference_scale_rad": 0.15},
        )
        self.rewards.crouch_foot_contact = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:crouch_foot_contact", weight=0.75,
            params={"command_name": "base_velocity", "sensor_cfg": crouch_contacts, "force_threshold_n": 5.0},
        )
        self.rewards.crouch_foot_contact_loss = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:crouch_foot_contact_loss", weight=-1.0,
            params={"command_name": "base_velocity", "sensor_cfg": crouch_contacts, "force_threshold_n": 5.0},
        )
        self.rewards.crouch_foot_slip = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:crouch_foot_slip", weight=-0.30,
            params={"command_name": "base_velocity", "asset_cfg": crouch_feet, "sensor_cfg": crouch_contacts},
        )
        self.rewards.crouch_action_rate = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:crouch_action_rate", weight=-0.01,
            params={"command_name": "base_velocity"},
        )
        self.rewards.crouch_joint_velocity_saturation = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:crouch_joint_saturation", weight=-0.12,
            params={"command_name": "base_velocity", "asset_cfg": crouch_sagittal, "quantity": "velocity", "threshold": 0.90, "scale": 0.10},
        )
        self.rewards.crouch_torque_saturation = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:crouch_joint_saturation", weight=-0.15,
            params={"command_name": "base_velocity", "asset_cfg": crouch_sagittal, "quantity": "torque", "threshold": 0.90, "scale": 0.10},
        )
        self.rewards.crouch_joint_limit = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:crouch_joint_limit_proximity", weight=-0.25,
            params={"command_name": "base_velocity", "asset_cfg": crouch_sagittal, "threshold": 0.85, "scale": 0.15},
        )
        self.rewards.crouch_hold_completion = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:crouch_completion", weight=0.5,
            params={"command_name": "base_velocity", "completion": "hold"},
        )
        self.rewards.crouch_return_completion = RewTerm(
            func="g1_command_skills.tasks.skill_rewards:crouch_completion", weight=0.75,
            params={"command_name": "base_velocity", "completion": "return"},
        )

        # Existing physical regularizers remain common safety terms.  The new
        # objective terms above are mutually gated and cross-faded for 0.4 s.
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.safe_periodic_flight.func = ResolvableString(
            "g1_command_skills.tasks.skill_rewards:GatedSafePeriodicFlightReward"
        )
        self.rewards.feet_slide.weight = -0.25


@configclass
class G1CommandRunEnvCfg(G1CommandBaseEnvCfg):
    command_mode: str = "run"


@configclass
class G1CommandTurnEnvCfg(G1CommandBaseEnvCfg):
    command_mode: str = "turn"

    def __post_init__(self) -> None:
        super().__post_init__()
        # First TURN curriculum: only +/-45 deg, low yaw rate, and extra time
        # to settle into the post-turn straight RUN segment.
        self.commands.base_velocity.turn_angles_deg = (45.0,)
        self.commands.base_velocity.turn_angle_probabilities = (1.0,)
        self.commands.base_velocity.turn_speed_range = (0.8, 1.4)
        self.commands.base_velocity.turn_script_durations_s = (1.5, 4.5, 3.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.45, 0.45)


@configclass
class G1CommandTurnFullEnvCfg(G1CommandTurnEnvCfg):
    """Second TURN curriculum, enabled only after the +/-45 deg gate passes."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.turn_angles_deg = (45.0, 90.0)
        # 90 deg is the new target; 45 deg remains rehearsal to preserve it.
        self.commands.base_velocity.turn_angle_probabilities = (0.30, 0.70)
        # The no-training model_0 audit found 100% left versus 80% right.
        self.commands.base_velocity.turn_direction_probabilities = (0.30, 0.70)
        self.commands.base_velocity.turn_script_durations_s = (1.5, 5.0, 3.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.75, 0.75)


@configclass
class G1CommandStopEnvCfg(G1CommandBaseEnvCfg):
    command_mode: str = "stop"

    def __post_init__(self) -> None:
        super().__post_init__()
        # The isolated STOP pilot begins with controlled forward momentum.
        # Sequence training reaches STOP through RUN and does not use this reset.
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.05, 0.05)
        self.events.reset_base.params["velocity_range"].update(
            {"x": (0.8, 1.4), "y": (-0.02, 0.02), "z": (0.0, 0.0), "yaw": (-0.05, 0.05)}
        )
        command = self.commands.base_velocity
        command.run_speed_range = (0.8, 1.4)
        command.stop_distance_range = (1.5, 2.5)
        command.stop_hold_duration_s = 1.0
        command.stop_script_durations_s = (2.0, 5.0)
        command.phase_duration_jitter_fraction = 0.0
        command.run_path_initial_lateral_error_range_m = (0.0, 0.0)


@configclass
class G1CommandStopBEnvCfg(G1CommandStopEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.run_speed_range = (1.4, 2.0)
        self.commands.base_velocity.stop_distance_range = (1.5, 3.0)
        self.events.reset_base.params["velocity_range"]["x"] = (1.4, 2.0)


@configclass
class G1CommandStopCEnvCfg(G1CommandStopEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.run_speed_range = (2.0, 2.6)
        self.commands.base_velocity.stop_distance_range = (2.0, 4.0)
        self.events.reset_base.params["velocity_range"]["x"] = (2.0, 2.6)


@configclass
class G1CommandCrouchEnvCfg(G1CommandBaseEnvCfg):
    """CROUCH Stage A: relative 8--15 cm depth from a stationary entry."""

    command_mode: str = "crouch"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 8.0
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.05, 0.05)
        self.events.reset_base.params["velocity_range"].update(
            {"x": (-0.03, 0.03), "y": (-0.02, 0.02), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-0.03, 0.03)}
        )
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        command = self.commands.base_velocity
        command.phase_duration_jitter_fraction = 0.0
        command.run_path_initial_lateral_error_range_m = (0.0, 0.0)
        command.ranges.lin_vel_x = (0.0, 0.0)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.25, 0.25)


@configclass
class G1CommandCrouchShallowEnvCfg(G1CommandCrouchEnvCfg):
    """Production shallow-CROUCH contract with no implicit deep-command clamp."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 10.0
        self.commands.base_velocity.crouch_height_drop_range_m = (0.08, 0.10)


@configclass
class G1CommandStepOverAuditEnvCfg(G1CommandCrouchShallowEnvCfg):
    """Quasi-static STEP_OVER reachability scene with the legacy 5 cm cuboid."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 14.0
        self.events.reset_base.params["pose_range"].update(
            {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        )
        self.scene.step_obstacle = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/StepObstacle",
            spawn=sim_utils.CuboidCfg(
                size=(0.06, 2.20, 0.05),
                collision_props=sim_schemas.CollisionPropertiesCfg(),
                rigid_props=sim_schemas.RigidBodyPropertiesCfg(kinematic_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.18, 0.08)),
                activate_contact_sensors=True,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.32, 0.0, 0.025)),
        )
        self.scene.step_obstacle_contact = ContactSensorCfg(
            prim_path="/World/envs/env_.*/StepObstacle",
            update_period=self.sim.dt,
            history_length=3,
            debug_vis=False,
        )


@configclass
class G1CommandSequenceEnvCfg(G1CommandBaseEnvCfg):
    command_mode: str = "sequence"


def _configure_play(cfg: G1CommandBaseEnvCfg) -> None:
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 2.5
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.commands.base_velocity.run_speed_range = (2.4, 2.4)
    cfg.commands.base_velocity.turn_speed_range = (2.0, 2.0)
    cfg.commands.base_velocity.stop_distance_range = (1.0, 1.0)
    cfg.commands.base_velocity.turn_angles_deg = (90.0,)
    cfg.commands.base_velocity.turn_angle_probabilities = (1.0,)
    cfg.commands.base_velocity.phase_duration_jitter_fraction = 0.0
    cfg.commands.base_velocity.run_path_initial_lateral_error_range_m = (0.0, 0.0)
    cfg.commands.base_velocity.rehearsal_probabilities = {
        "run": (1.0, 0.0, 0.0, 0.0, 0.0),
        "turn": (0.0, 1.0, 0.0, 0.0, 0.0),
        "stop": (0.0, 0.0, 1.0, 0.0, 0.0),
        "sequence": (0.0, 0.0, 0.0, 1.0, 0.0),
        "crouch": (0.0, 0.0, 0.0, 0.0, 1.0),
    }[cfg.command_mode]


@configclass
class G1CommandRunPlayEnvCfg(G1CommandRunEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_play(self)


@configclass
class G1CommandTurnPlayEnvCfg(G1CommandTurnEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_play(self)


@configclass
class G1CommandTurnFullPlayEnvCfg(G1CommandTurnFullEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_play(self)


@configclass
class G1CommandStopPlayEnvCfg(G1CommandStopEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_play(self)
        # STOP play/eval is the Stage-A distribution, not the historical
        # 2.4 m/s / 1.0 m emergency-stop probe used by other play modes.
        self.commands.base_velocity.run_speed_range = (0.8, 1.4)
        self.commands.base_velocity.stop_distance_range = (1.5, 2.5)


@configclass
class G1CommandStopBPlayEnvCfg(G1CommandStopBEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_play(self)
        self.commands.base_velocity.run_speed_range = (1.4, 2.0)
        self.commands.base_velocity.stop_distance_range = (1.5, 3.0)


@configclass
class G1CommandStopCPlayEnvCfg(G1CommandStopCEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_play(self)
        self.commands.base_velocity.run_speed_range = (2.0, 2.6)
        self.commands.base_velocity.stop_distance_range = (2.0, 4.0)


@configclass
class G1CommandCrouchPlayEnvCfg(G1CommandCrouchEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_play(self)
        self.commands.base_velocity.rehearsal_probabilities = (0.0, 0.0, 0.0, 0.0, 1.0)


@configclass
class G1CommandCrouchShallowPlayEnvCfg(G1CommandCrouchShallowEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_play(self)
        self.commands.base_velocity.rehearsal_probabilities = (0.0, 0.0, 0.0, 0.0, 1.0)


@configclass
class G1CommandStepOverAuditPlayEnvCfg(G1CommandStepOverAuditEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_play(self)
        self.commands.base_velocity.rehearsal_probabilities = (0.0, 0.0, 0.0, 0.0, 1.0)


@configclass
class G1CommandSequencePlayEnvCfg(G1CommandSequenceEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_play(self)


@configclass
class G1CommandRunEvalEnvCfg(G1CommandRunPlayEnvCfg):
    pass


@configclass
class G1CommandTurnEvalEnvCfg(G1CommandTurnPlayEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        command = self.commands.base_velocity
        command.turn_angles_deg = (45.0,)
        command.turn_angle_probabilities = (1.0,)
        command.turn_script_durations_s = (1.5, 4.5, 3.5)
        command.deterministic_turn_evaluation = True


@configclass
class G1CommandTurnFullEvalEnvCfg(G1CommandTurnEvalEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.turn_angles_deg = (45.0, 90.0)
        self.commands.base_velocity.turn_angle_probabilities = (0.5, 0.5)
        self.commands.base_velocity.turn_script_durations_s = (1.5, 5.0, 3.5)
        # Match TurnFull training.  This class inherits TurnEval (the 45-deg
        # curriculum), so the full-range override must be repeated explicitly.
        self.commands.base_velocity.ranges.ang_vel_z = (-0.75, 0.75)


@configclass
class G1CommandStopEvalEnvCfg(G1CommandStopPlayEnvCfg):
    pass


@configclass
class G1CommandStopBEvalEnvCfg(G1CommandStopBPlayEnvCfg):
    pass


@configclass
class G1CommandStopCEvalEnvCfg(G1CommandStopCPlayEnvCfg):
    pass


@configclass
class G1CommandCrouchEvalEnvCfg(G1CommandCrouchPlayEnvCfg):
    pass


@configclass
class G1CommandCrouchShallowEvalEnvCfg(G1CommandCrouchShallowPlayEnvCfg):
    pass


@configclass
class G1CommandStepOverAuditEvalEnvCfg(G1CommandStepOverAuditPlayEnvCfg):
    pass


@configclass
class G1CommandSequenceEvalEnvCfg(G1CommandSequencePlayEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.turn_angles_deg = (45.0, 90.0)
        self.commands.base_velocity.turn_angle_probabilities = (0.5, 0.5)
        self.commands.base_velocity.deterministic_turn_evaluation = True
