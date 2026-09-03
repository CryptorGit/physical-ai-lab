"""Load and validate the frozen OpenDuckMini hardware-safe training contract.

The JSON file is the single data source for the overlay.  Validation happens at
import time so a malformed or partially edited contract fails before training.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


CONTRACT_PATH = Path(__file__).resolve().parents[1] / "contract.json"
FORMAL_LEG_TARGET_MARGIN_RAD = 0.050
FORMAL_TARGET_SLEW_LIMIT_RAD_PER_S = 2.0
FORMAL_RESET_NOISE_MARGIN_RAD = 0.005
FORMAL_CONTROL_FIRST_STARTUP_DT_S = 0.02
FORMAL_MAX_TARGET_DELTA_PER_TICK_RAD = 0.04
FORMAL_BACKWARD_EXIT_RECOVERY_STATUS = "ADOPTED_SIMULATION_ONLY"
FORMAL_BACKWARD_EXIT_RECOVERY_CANDIDATE_EVIDENCE_SHA256 = (
    "f040a9c6f9783b7d50dd5590389d3c81411e8f3a7fa9dd155e8ac78175d5ff56"
)
FORMAL_BACKWARD_EXIT_RECOVERY_SUPERSEDED_H2_SELECTION_EVIDENCE_SHA256 = (
    "6f65bef5053da5962442eca3bf46b855a36691aa9bbad84496c9892b36ee0de4"
)
FORMAL_BACKWARD_EXIT_RECOVERY_SUPERSEDED_H2_ADOPTION_EVIDENCE_SHA256 = (
    "bd7e8a79b32880fa63e54570854682b5b8912f1cdafeed8e80273501dc6ef611"
)
FORMAL_BACKWARD_EXIT_RECOVERY_ADOPTION_EVIDENCE_SHA256 = (
    "1aea58904598cfba8ea4ef572f9473bba647eacc695f7fce3fcaa1b8646391aa"
)
FORMAL_BACKWARD_EXIT_RECOVERY_SAFETY_COMPONENT_EVIDENCE_SHA256 = (
    "090e09cc2d82c1f42112a5f30a85cd93d940213956d6ec902fb4089875fb855a"
)
FORMAL_BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD = 0.0225
FORMAL_BACKWARD_EXIT_RECOVERY_HOLD_TICKS = 13
FORMAL_BACKWARD_EXIT_RECOVERY_HOLD_SECONDS = 0.26
FORMAL_BACKWARD_EXIT_RECOVERY_LEFT_KNEE_SAFE_UPPER_RAD = 0.475534
FORMAL_BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD = 0.403034
FORMAL_CONTROL_FIRST_STARTUP_ORDER = (
    "reset_qpos_and_guard_state",
    "observe_route_and_infer_first_command_policy",
    "compose_first_desired_joint_targets",
    "guard.control_first_startup(first_desired_targets, dt=0.02)",
    "apply_guarded_targets_to_actuators",
    "first_physics_step",
    "first_post_step_sensor_sample",
)
FORMAL_NORMAL_TICK_ORDER = (
    "observe_route_and_infer_policy",
    "compose_desired_joint_targets",
    "clamp_desired_leg_targets_to_inward_margin",
    "slew_leg_targets_exactly_once_from_previous_applied_state",
    "final_leg_target_physical_safe_clamp",
    "apply_guarded_targets_to_actuators",
    "physics_step",
    "post_step_sensor_and_audit",
)


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Read and validate a contract file, returning its decoded contents."""

    contract = json.loads(path.read_text(encoding="utf-8"))
    validate_contract(contract)
    return contract


def _require_exact_keys(
    mapping: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} key mismatch: missing={missing}, extra={extra}")


def validate_contract(contract: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when a safety invariant is not explicit and true."""

    if contract.get("schema_version") != 1:
        raise ValueError("unsupported contract schema")

    joint_order = tuple(contract["actuator_joint_order"])
    if len(joint_order) != 14 or len(set(joint_order)) != 14:
        raise ValueError("actuator_joint_order must contain 14 unique joints")
    joint_set = set(joint_order)

    head_joints = set(contract["head_joint_names"])
    if head_joints != {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}:
        raise ValueError("the four head joints must be named explicitly")

    calibration = contract["calibration"]
    for key in ("zero_raw", "zero_offset_rad", "joint_direction"):
        _require_exact_keys(calibration[key], joint_set, f"calibration.{key}")

    encoder_steps = int(contract["provenance"]["encoder_steps"])
    for name in joint_order:
        raw = int(calibration["zero_raw"][name])
        expected_offset = (raw - encoder_steps / 2) * (
            2 * math.pi / encoder_steps
        )
        actual_offset = float(calibration["zero_offset_rad"][name])
        if not math.isclose(actual_offset, expected_offset, abs_tol=1e-12):
            raise ValueError(f"{name} zero offset does not match raw count")
        if float(calibration["joint_direction"][name]) not in (-1.0, 1.0):
            raise ValueError(f"{name} direction must be -1 or +1")

    expected_negative = {"left_knee", "left_ankle"}
    actual_negative = {
        name
        for name, direction in calibration["joint_direction"].items()
        if float(direction) < 0.0
    }
    if actual_negative != expected_negative:
        raise ValueError("only left_knee and left_ankle may have negative direction")

    safe_init = contract["safe_init_pos_rad"]
    _require_exact_keys(safe_init, joint_set, "safe_init_pos_rad")
    limits = contract["safe_joint_limits_rad"]
    leg_joints = joint_set - head_joints
    _require_exact_keys(limits, leg_joints, "safe_joint_limits_rad")
    for name, (lower, upper) in limits.items():
        value = float(safe_init[name])
        if not float(lower) <= value <= float(upper):
            raise ValueError(f"{name} SAFE_INIT is outside the safe limit")

    target_safety = contract["target_safety"]
    target_margin = float(target_safety["leg_margin_rad"])
    if not math.isfinite(target_margin) or target_margin <= 0.0:
        raise ValueError("target_safety.leg_margin_rad must be finite and positive")
    if target_margin != FORMAL_LEG_TARGET_MARGIN_RAD:
        raise ValueError("target_safety.leg_margin_rad must remain exactly 0.050")
    if float(target_safety["head_target_rad"]) != 0.0:
        raise ValueError("target_safety.head_target_rad must be exactly zero")
    target_slew_limit = float(target_safety["target_slew_limit_rad_per_s"])
    if not math.isfinite(target_slew_limit) or target_slew_limit <= 0.0:
        raise ValueError(
            "target_safety.target_slew_limit_rad_per_s must be finite and positive"
        )
    if target_slew_limit != FORMAL_TARGET_SLEW_LIMIT_RAD_PER_S:
        raise ValueError(
            "target_safety.target_slew_limit_rad_per_s must remain exactly 2.0"
        )
    for name, (lower, upper) in limits.items():
        if float(lower) + target_margin > float(upper) - target_margin:
            raise ValueError(f"{name} is too narrow for the target safety margin")
    for key in ("rule", "startup_transition", "qpos_acceptance"):
        if not target_safety.get(key):
            raise ValueError(f"target_safety.{key} must be explicit")
    control_first = target_safety["control_first_startup"]
    reset_qpos = target_safety["reset_qpos"]
    reset_noise_margin = float(reset_qpos["noise_margin_rad"])
    if not math.isfinite(reset_noise_margin) or reset_noise_margin <= 0.0:
        raise ValueError("reset_qpos.noise_margin_rad must be finite and positive")
    if reset_noise_margin != FORMAL_RESET_NOISE_MARGIN_RAD:
        raise ValueError("reset_qpos.noise_margin_rad must remain exactly 0.005")
    for name, (lower, upper) in limits.items():
        if float(lower) + reset_noise_margin > float(upper) - reset_noise_margin:
            raise ValueError(f"{name} is too narrow for the reset noise margin")
    for key in ("zero_noise_rule", "positive_noise_rule", "head_rule"):
        if not reset_qpos.get(key):
            raise ValueError(f"target_safety.reset_qpos.{key} must be explicit")
    if not math.isclose(
        float(safe_init["left_knee"]),
        float(limits["left_knee"][1]) - reset_noise_margin,
        abs_tol=1e-12,
    ):
        raise ValueError("left_knee SAFE_INIT must equal upper SAFE minus reset margin")

    startup = control_first
    if startup.get("required") is not True:
        raise ValueError("control-first startup must be required")
    startup_dt = float(startup.get("control_dt_seconds", -1.0))
    if not math.isfinite(startup_dt) or startup_dt != FORMAL_CONTROL_FIRST_STARTUP_DT_S:
        raise ValueError("control-first startup dt must remain exactly 0.02")
    if startup.get("desired_targets") != "first_command_policy_targets":
        raise ValueError("startup must guard the first command-policy targets")
    if startup.get("home_only_precharge") != "PROHIBITED":
        raise ValueError("home-only startup precharge must be prohibited")
    if (
        int(startup.get("physics_steps_allowed_before_control", -1)) != 0
        or int(startup.get("physics_steps_before_guarded_control", -1)) != 0
    ):
        raise ValueError(
            "control-first guarded control permits no physics steps before "
            "control and must precede all physics"
        )
    if int(startup.get("guard_steps_before_first_physics", -1)) != 1:
        raise ValueError("startup requires exactly one guard step before physics")
    if int(startup.get("slew_applications_per_tick", -1)) != 1:
        raise ValueError("target slew must be applied exactly once per tick")
    if float(startup.get("maximum_leg_target_delta_per_tick_rad", -1.0)) != (
        FORMAL_MAX_TARGET_DELTA_PER_TICK_RAD
    ):
        raise ValueError("maximum leg target delta per tick must remain 0.04 rad")
    for key in ("desired_target_rule", "audit_required"):
        if not startup.get(key):
            raise ValueError(f"target_safety.control_first_startup.{key} is required")
    if tuple(startup.get("required_order", ())) != FORMAL_CONTROL_FIRST_STARTUP_ORDER:
        raise ValueError("control-first startup order mismatch")
    if tuple(startup.get("normal_tick_order", ())) != FORMAL_NORMAL_TICK_ORDER:
        raise ValueError("normal control tick order mismatch")

    recovery = target_safety.get("backward_exit_recovery")
    if not isinstance(recovery, Mapping):
        raise ValueError("backward-exit recovery contract is missing")
    if recovery.get("status") != FORMAL_BACKWARD_EXIT_RECOVERY_STATUS:
        raise ValueError("backward-exit recovery adoption status mismatch")
    if recovery.get("enabled_by_default") is not True:
        raise ValueError("backward-exit recovery must be enabled for H3")
    if recovery.get("formal_candidate_only") is not False:
        raise ValueError("adopted backward-exit recovery cannot remain candidate-only")
    if recovery.get("diagnostic_unadopted_only") is not False:
        raise ValueError("adopted recovery is not the diagnostic history path")
    if recovery.get("adoption_eligible") is not True:
        raise ValueError("H3 recovery adoption eligibility must remain enabled")
    if recovery.get("simulation_acceptance_eligible") is not True:
        raise ValueError("H3 recovery simulation eligibility must remain enabled")
    if recovery.get("candidate_selection_evidence_sha256") != (
        FORMAL_BACKWARD_EXIT_RECOVERY_CANDIDATE_EVIDENCE_SHA256
    ):
        raise ValueError("backward-exit recovery candidate evidence hash mismatch")
    if recovery.get("superseded_h2_selection_evidence_sha256") != (
        FORMAL_BACKWARD_EXIT_RECOVERY_SUPERSEDED_H2_SELECTION_EVIDENCE_SHA256
    ):
        raise ValueError("superseded H2 selection evidence hash mismatch")
    if recovery.get("superseded_h2_adoption_evidence_sha256") != (
        FORMAL_BACKWARD_EXIT_RECOVERY_SUPERSEDED_H2_ADOPTION_EVIDENCE_SHA256
    ):
        raise ValueError("superseded H2 adoption evidence hash mismatch")
    if recovery.get("adoption_evidence_sha256") != (
        FORMAL_BACKWARD_EXIT_RECOVERY_ADOPTION_EVIDENCE_SHA256
    ):
        raise ValueError("H3 adoption evidence hash mismatch")
    if recovery.get("safety_component_evidence_sha256") != (
        FORMAL_BACKWARD_EXIT_RECOVERY_SAFETY_COMPONENT_EVIDENCE_SHA256
    ):
        raise ValueError("H3 recovery safety-component evidence hash mismatch")
    if recovery.get("safety_component_only") is not False:
        raise ValueError("passing H3 5x15 selection is not safety-component-only")
    if recovery.get("safety_component_evidence_is_safety_only") is not True:
        raise ValueError("H3 safety component must remain explicitly safety-only")
    if recovery.get("fast_exit_safety_passed") is not True:
        raise ValueError("H3 recovery fast-exit safety must be explicit")
    if recovery.get("combined_5x15_required") is not False:
        raise ValueError("passing H3 recovery must not require another combined 5x15")
    if recovery.get("combined_5x15_passed") is not True:
        raise ValueError("H3 recovery combined 5x15 pass must be explicit")
    if recovery.get("requires_formal_20x30_requalification") is not False:
        raise ValueError("adopted H3 recovery is already 20x30 requalified")
    expected_recovery_strings = {
        "activation": "backward_feedforward_active_true_to_false",
        "joint_name": "left_knee",
        "release": "instant_after_hold",
        "backward_reentry": "cancel_remaining_recovery",
        "reset": "clear_active_state_and_history",
        "composition_stage": (
            "after_policy_or_profile_before_final_target_guard"
        ),
        "hardware_deployment": "PROHIBITED",
    }
    for key, expected in expected_recovery_strings.items():
        if recovery.get(key) != expected:
            raise ValueError(f"backward-exit recovery {key} mismatch")
    if recovery.get("exit_tick_is_first_active_tick") is not True:
        raise ValueError("backward-exit recovery exit tick must be active")
    if int(recovery.get("joint_index", -1)) != joint_order.index("left_knee"):
        raise ValueError("backward-exit recovery left-knee index mismatch")
    if int(recovery.get("final_guard_calls_per_tick", -1)) != 1:
        raise ValueError("backward-exit recovery requires one final guard call")
    if float(recovery.get("safe_upper_rad", math.nan)) != (
        FORMAL_BACKWARD_EXIT_RECOVERY_LEFT_KNEE_SAFE_UPPER_RAD
    ):
        raise ValueError("backward-exit recovery SAFE upper mismatch")
    if float(recovery.get("base_target_margin_rad", math.nan)) != target_margin:
        raise ValueError("backward-exit recovery base target margin mismatch")
    if float(recovery.get("extra_upper_margin_rad", math.nan)) != (
        FORMAL_BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD
    ):
        raise ValueError("backward-exit recovery extra upper margin mismatch")
    if float(recovery.get("upper_target_rad", math.nan)) != (
        FORMAL_BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD
    ):
        raise ValueError("backward-exit recovery upper target mismatch")
    if int(recovery.get("hold_control_ticks", -1)) != (
        FORMAL_BACKWARD_EXIT_RECOVERY_HOLD_TICKS
    ):
        raise ValueError("backward-exit recovery hold ticks mismatch")
    if float(recovery.get("control_dt_seconds", math.nan)) != startup_dt:
        raise ValueError("backward-exit recovery control dt mismatch")
    if float(recovery.get("hold_seconds", math.nan)) != (
        FORMAL_BACKWARD_EXIT_RECOVERY_HOLD_SECONDS
    ):
        raise ValueError("backward-exit recovery hold seconds mismatch")
    if not math.isclose(
        float(recovery["upper_target_rad"]),
        float(limits["left_knee"][1])
        - target_margin
        - float(recovery["extra_upper_margin_rad"]),
        abs_tol=1e-12,
    ):
        raise ValueError("backward-exit recovery upper target is not inward-safe")
    if not math.isclose(
        float(recovery["hold_seconds"]),
        int(recovery["hold_control_ticks"])
        * float(recovery["control_dt_seconds"]),
        abs_tol=1e-12,
    ):
        raise ValueError("backward-exit recovery hold duration mismatch")

    head_lock = contract["head_lock"]
    if head_lock.get("enabled") is not True:
        raise ValueError("head lock must be enabled")
    for key in ("command_rad", "action_residual_rad", "qpos_noise_scale_rad"):
        if float(head_lock[key]) != 0.0:
            raise ValueError(f"head_lock.{key} must be exactly zero")
    if float(contract["qpos_noise_scale_rad"]["head"]) != 0.0:
        raise ValueError("qpos_noise_scale_rad.head must be exactly zero")
    if any(float(safe_init[name]) != 0.0 for name in head_joints):
        raise ValueError("all SAFE_INIT head positions must be exactly zero")

    if contract["deployment"].get("hardware_status") != "PROHIBITED":
        raise ValueError("hardware deployment must remain prohibited")
    if not contract["deployment"].get("required_gates"):
        raise ValueError("hardware deployment gates must be explicit")


CONTRACT = load_contract()
ACTUATOR_JOINT_ORDER = tuple(CONTRACT["actuator_joint_order"])
HEAD_JOINTS = frozenset(CONTRACT["head_joint_names"])
LEG_JOINTS = frozenset(set(ACTUATOR_JOINT_ORDER) - HEAD_JOINTS)
SAFE_INIT_POS = {
    name: float(value) for name, value in CONTRACT["safe_init_pos_rad"].items()
}
SAFE_JOINT_LIMITS = {
    name: (float(bounds[0]), float(bounds[1]))
    for name, bounds in CONTRACT["safe_joint_limits_rad"].items()
}
LEG_TARGET_MARGIN_RAD = float(CONTRACT["target_safety"]["leg_margin_rad"])
TARGET_SLEW_LIMIT_RAD_PER_S = float(
    CONTRACT["target_safety"]["target_slew_limit_rad_per_s"]
)
CONTROL_FIRST_STARTUP_DT_S = float(
    CONTRACT["target_safety"]["control_first_startup"]["control_dt_seconds"]
)
RESET_NOISE_MARGIN_RAD = float(
    CONTRACT["target_safety"]["reset_qpos"]["noise_margin_rad"]
)
BACKWARD_EXIT_RECOVERY_CONTRACT = dict(
    CONTRACT["target_safety"]["backward_exit_recovery"]
)
BACKWARD_EXIT_RECOVERY_STATUS = str(
    BACKWARD_EXIT_RECOVERY_CONTRACT["status"]
)
BACKWARD_EXIT_RECOVERY_CANDIDATE_EVIDENCE_SHA256 = str(
    BACKWARD_EXIT_RECOVERY_CONTRACT["candidate_selection_evidence_sha256"]
)
BACKWARD_EXIT_RECOVERY_SUPERSEDED_H2_SELECTION_EVIDENCE_SHA256 = str(
    BACKWARD_EXIT_RECOVERY_CONTRACT[
        "superseded_h2_selection_evidence_sha256"
    ]
)
BACKWARD_EXIT_RECOVERY_ADOPTION_EVIDENCE_SHA256 = str(
    BACKWARD_EXIT_RECOVERY_CONTRACT["adoption_evidence_sha256"]
)
BACKWARD_EXIT_RECOVERY_SUPERSEDED_H2_ADOPTION_EVIDENCE_SHA256 = str(
    BACKWARD_EXIT_RECOVERY_CONTRACT[
        "superseded_h2_adoption_evidence_sha256"
    ]
)
BACKWARD_EXIT_RECOVERY_SAFETY_COMPONENT_EVIDENCE_SHA256 = str(
    BACKWARD_EXIT_RECOVERY_CONTRACT["safety_component_evidence_sha256"]
)
BACKWARD_EXIT_RECOVERY_ENABLED_BY_DEFAULT = bool(
    BACKWARD_EXIT_RECOVERY_CONTRACT["enabled_by_default"]
)
BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD = float(
    BACKWARD_EXIT_RECOVERY_CONTRACT["extra_upper_margin_rad"]
)
BACKWARD_EXIT_RECOVERY_HOLD_TICKS = int(
    BACKWARD_EXIT_RECOVERY_CONTRACT["hold_control_ticks"]
)
BACKWARD_EXIT_RECOVERY_HOLD_SECONDS = float(
    BACKWARD_EXIT_RECOVERY_CONTRACT["hold_seconds"]
)
BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD = float(
    BACKWARD_EXIT_RECOVERY_CONTRACT["upper_target_rad"]
)
