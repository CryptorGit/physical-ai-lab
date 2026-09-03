"""Evaluation schema and deterministic failure classification."""

from __future__ import annotations

from dataclasses import dataclass


FAILURE_CLASSES = (
    "tracking_failure",
    "heading_failure",
    "overshoot",
    "stop_failure",
    "course_deviation",
    "insufficient_crouch_depth",
    "excessive_crouch_depth",
    "crouch_hold_failure",
    "unstable_crouch",
    "return_failure",
    "standing_settle_failure",
    "unsupported_crouch_depth",
    "base_transition_failure",
    "stand_hold_failure",
    "foot_contact_loss",
    "both_feet_airborne",
    "support_foot_loss",
    "prolonged_single_support",
    "unstable_contact_switching",
    "support_shift_failure",
    "lead_lift_failure",
    "lead_insufficient_clearance",
    "lead_front_collision",
    "lead_top_collision",
    "lead_placement_failure",
    "weight_transfer_failure",
    "trail_lift_failure",
    "trail_insufficient_clearance",
    "trail_front_collision",
    "trail_top_collision",
    "trail_placement_failure",
    "unstable_recovery",
    "unsupported_obstacle",
    "single_step_unavailable",
    "desired_lead_phase_unavailable",
    "step_liftoff_failure",
    "step_clearance_failure",
    "step_placement_failure",
    "touchdown_failure",
    "double_support_recovery_failure",
    "zero_command_recovery_failure",
    "obstacle_distance_unreachable",
    "lead_step_collision",
    "trail_step_collision",
    "invalid_initial_clearance",
    "asymmetric_initial_pose",
    "premature_contact",
    "excessive_contact_asymmetry",
    "impact_failure",
    "absorption_failure",
    "double_support_failure",
    "excessive_rebound",
    "stand_hold_failure",
    "joint_limit_failure",
    "foot_slip_failure",
    "unsupported_drop_height",
    "obstacle_collision",
    "insufficient_clearance",
    "unstable_landing",
    "recovery_failure",
    "saturation_failure",
    "fall",
    "timeout",
)


@dataclass(frozen=True)
class SkillThresholds:
    speed_error_mps: float = 0.25
    heading_error_rad: float = 0.12
    stop_position_error_m: float = 0.50
    stop_speed_mps: float = 0.20
    course_deviation_m: float = 0.75
    tilt_rad: float = 0.20
    joint_velocity_saturation_fraction: float = 0.05
    ankle_torque_saturation_fraction: float = 0.20
    torque_saturation_fraction: float = 0.05
    crouch_depth_error_m: float = 0.04
    crouch_return_height_error_m: float = 0.05


def classify_failure(record: dict, thresholds: SkillThresholds = SkillThresholds()) -> str:
    """Return the first actionable failure in a stable priority order."""
    if bool(record.get("fall", False)):
        return "fall"
    if float(record.get("joint_velocity_saturation_fraction", 0.0)) > thresholds.joint_velocity_saturation_fraction:
        return "saturation_failure"
    if float(record.get("ankle_torque_saturation_fraction", 0.0)) > thresholds.ankle_torque_saturation_fraction:
        return "saturation_failure"
    if float(record.get("torque_saturation_fraction", 0.0)) > thresholds.torque_saturation_fraction:
        return "saturation_failure"
    skill = str(record.get("skill", "")).upper()
    if skill == "RUN":
        if not bool(record.get("periodic_running", False)):
            return "tracking_failure"
        if float(record.get("speed_error_mps", 0.0)) > thresholds.speed_error_mps:
            return "tracking_failure"
        if float(record.get("heading_error_rad", 0.0)) > thresholds.heading_error_rad:
            return "heading_failure"
    elif skill == "TURN":
        if float(record.get("final_turn_angle_error_rad", float("inf"))) > thresholds.heading_error_rad:
            return "heading_failure"
        if not bool(record.get("straight_recovery_success", False)):
            return "recovery_failure"
        if float(record.get("speed_error_mps", 0.0)) > thresholds.speed_error_mps:
            return "tracking_failure"
    elif skill == "STOP":
        if not bool(record.get("stop_hold_success", False)):
            return "stop_failure"
        if float(record.get("heading_error_rad", 0.0)) > thresholds.heading_error_rad:
            return "heading_failure"
        if float(record.get("stop_position_error_m", 0.0)) > thresholds.stop_position_error_m:
            return "overshoot" if float(record.get("stop_target_longitudinal_m", 0.0)) < -thresholds.stop_position_error_m else "stop_failure"
        if float(record.get("stop_speed_mps", 0.0)) > thresholds.stop_speed_mps:
            return "stop_failure"
    elif skill == "CROUCH":
        if not bool(record.get("settle_success", True)):
            return "standing_settle_failure"
        if bool(record.get("base_transition_failure", False)):
            return "base_transition_failure"
        for field, failure in (
            ("both_feet_airborne_failure", "both_feet_airborne"),
            ("support_foot_loss_failure", "support_foot_loss"),
            ("prolonged_single_support_failure", "prolonged_single_support"),
            ("unstable_contact_switching_failure", "unstable_contact_switching"),
        ):
            if bool(record.get(field, False)):
                return failure
        if float(record.get("tilt_rad", 0.0)) > thresholds.tilt_rad:
            return "unstable_crouch"
        depth_error = float(record.get("actual_height_drop_m", 0.0)) - float(
            record.get("commanded_height_drop_m", 0.0)
        )
        if depth_error < -thresholds.crouch_depth_error_m:
            return "insufficient_crouch_depth"
        if depth_error > thresholds.crouch_depth_error_m:
            return "excessive_crouch_depth"
        if not bool(record.get("crouch_hold_success", False)):
            return "crouch_hold_failure"
        if not bool(record.get("return_to_stand_success", False)) or abs(
            float(record.get("return_height_error_m", float("inf")))
        ) > thresholds.crouch_return_height_error_m:
            return "return_failure"
        if not bool(record.get("stand_hold_success", record.get("return_to_stand_success", False))):
            return "stand_hold_failure"
    elif skill == "STEP_OVER":
        if bool(record.get("obstacle_collision", False)):
            return "obstacle_collision"
        if float(record.get("clearance_m", 1.0)) < 0.03:
            return "insufficient_clearance"
    elif skill == "LAND":
        if float(record.get("tilt_rad", 0.0)) > thresholds.tilt_rad:
            return "unstable_landing"
        if float(record.get("recovery_time_s", 0.0)) > 2.0:
            return "recovery_failure"
    if float(record.get("course_deviation_m", 0.0)) > thresholds.course_deviation_m:
        return "course_deviation"
    if bool(record.get("timeout", False)):
        return "timeout"
    return ""


def skill_success(record: dict, thresholds: SkillThresholds = SkillThresholds()) -> bool:
    return classify_failure(record, thresholds) == ""
