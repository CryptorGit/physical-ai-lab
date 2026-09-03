"""Canonical feature slices and leakage-ablation conditions."""

from __future__ import annotations

TIMING_INDICES = (145, 146, 147, 148)
LEGACY_DIM = 123
COMMAND_DIM = 29
OBSERVATION_DIM = 152
ACTION_DIM = 37

LEGACY_FIELDS = [
    {"name": "base_linear_velocity", "start": 0, "end": 3},
    {"name": "base_angular_velocity", "start": 3, "end": 6},
    {"name": "projected_gravity", "start": 6, "end": 9},
    {"name": "motion_command", "start": 9, "end": 12},
    {"name": "joint_position", "start": 12, "end": 49},
    {"name": "joint_velocity", "start": 49, "end": 86},
    {"name": "global_previous_action", "start": 86, "end": 123},
]

COMMAND_FIELDS = [
    {"name": "current_skill_one_hot", "start": 123, "end": 129},
    {"name": "previous_skill_one_hot", "start": 129, "end": 135},
    {"name": "sin_target_heading_error", "start": 135, "end": 136},
    {"name": "cos_target_heading_error", "start": 136, "end": 137},
    {"name": "skill_local_target_state", "start": 137, "end": 139},
    {"name": "relative_target_pelvis_height", "start": 139, "end": 140},
    {"name": "skill_local_auxiliary_state", "start": 140, "end": 144},
    {"name": "target_vertical_velocity", "start": 144, "end": 145},
    {"name": "normalized_elapsed_time", "start": 145, "end": 146},
    {"name": "normalized_remaining_time", "start": 146, "end": 147},
    {"name": "skill_phase", "start": 147, "end": 148},
    {"name": "transition_progress", "start": 148, "end": 149},
    {"name": "recovery_mode", "start": 149, "end": 150},
    {"name": "target_posture_roll_pitch", "start": 150, "end": 152},
]

ANALYSIS_PHASE_FIELDS = (
    "left_contact",
    "right_contact",
    "left_foot_air_time",
    "right_foot_air_time",
    "last_landing_foot",
    "support_phase",
    "left_contact_force",
    "right_contact_force",
)

FEATURE_CONDITIONS = {
    "A_full_152D": {"observation_indices": list(range(152)), "include_action": False, "include_phase": False},
    "B_152D_without_timing": {
        "observation_indices": [index for index in range(152) if index not in TIMING_INDICES],
        "include_action": False,
        "include_phase": False,
    },
    "C_legacy_123D": {"observation_indices": list(range(123)), "include_action": False, "include_phase": False},
    "D_legacy_123D_plus_action": {"observation_indices": list(range(123)), "include_action": True, "include_phase": False},
    "E_explicit_phase_upper_bound": {"observation_indices": list(range(152)), "include_action": False, "include_phase": True},
}


def feature_layout_document() -> dict:
    return {
        "observation_dimension": OBSERVATION_DIM,
        "legacy_dimension": LEGACY_DIM,
        "command_dimension": COMMAND_DIM,
        "action_dimension": ACTION_DIM,
        "legacy_fields": LEGACY_FIELDS,
        "command_fields": COMMAND_FIELDS,
        "timing_leakage_absolute_indices": list(TIMING_INDICES),
        "analysis_only_phase_fields": list(ANALYSIS_PHASE_FIELDS),
        "feature_conditions": FEATURE_CONDITIONS,
        "production_observation_modified": False,
    }
