"""Fast checks that do not require launching Isaac Sim."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    source_root = ROOT / "src/g1_command_skills"
    files = sorted(source_root.rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    command_source = (source_root / "tasks/command_mdp.py").read_text(encoding="utf-8")
    env_source = (source_root / "tasks/g1_command_env_cfg.py").read_text(encoding="utf-8")
    failures_source = (source_root / "evaluation.py").read_text(encoding="utf-8")
    actor_source = (source_root / "models/residual_actor.py").read_text(encoding="utf-8")
    reward_source = (source_root / "tasks/skill_rewards.py").read_text(encoding="utf-8")
    evaluation_source = (ROOT / "scripts/evaluate.py").read_text(encoding="utf-8")
    result = {
        "python_files_parsed": len(files),
        "legacy_observation_dim": 123,
        "extra_command_dim": 29,
        "policy_observation_dim": 152,
        "legacy_prefix_preserved": "LEGACY_OBSERVATION_DIM = 123" in command_source,
        "sequence_declared": "SkillId.RUN, SkillId.TURN, SkillId.RUN, SkillId.STOP" in command_source,
        "command_observation_appended": "policy.motion_command" in env_source,
        "failure_classes_declared": all(
            name in failures_source for name in (
                "tracking_failure", "heading_failure", "overshoot", "stop_failure",
                "course_deviation", "insufficient_crouch_depth", "excessive_crouch_depth",
                "crouch_hold_failure", "unstable_crouch", "return_failure", "foot_contact_loss",
                "obstacle_collision", "insufficient_clearance", "unstable_landing",
                "recovery_failure", "saturation_failure", "fall", "timeout",
            )
        ),
        "frozen_base_actor": "parameter.requires_grad_(False)" in actor_source,
        "bounded_skill_residuals": "else self.residual_scale) * torch.tanh" in actor_source,
        "bounded_stop_correction": "self.stop_correction_scale * torch.tanh" in actor_source,
        "parent_action_preserved": "parent_action_mean + total_stop_correction" in actor_source,
        "corrective_rewards_declared": all(
            value in reward_source for value in (
                "stop_heading_tail", "stop_yaw_rate_tracking", "stop_attitude_stability",
                "stop_instability_tail", "stop_hold_heading", "stop_parent_action_deviation",
            )
        ),
        "command_gate_declared": "transition * current_gate" in actor_source,
        "skill_local_learnable_routes": all(
            value in actor_source for value in (
                "skill_command_encoders", "skill_state_adapters", "_expand_legacy_shared_routes"
            )
        ),
        "rehearsal_declared": all(value in env_source for value in ("(0.30, 0.70", "(0.20, 0.20, 0.60", "(0.10, 0.10, 0.10, 0.70")),
        "path_command_reuses_29_columns": all(
            value in command_source for value in (
                "path_lateral_error", "path_forward_velocity", "path_lateral_velocity", "path_curvature"
            )
        ),
        "path_rewards_declared": all(
            value in reward_source for value in (
                "run_path_lateral_error", "run_path_lateral_velocity", "RunPathRecoveryProgressReward"
            )
        ),
        "run_evaluation_fields_declared": all(
            value in evaluation_source for value in (
                "skill_success_rate", "course_deviation_failure_rate", "first_failure_reason",
                "path_lateral_error_p95", "residual_action_norm"
            )
        ),
        "turn_accumulated_yaw_declared": all(
            value in command_source for value in (
                "turn_start_heading_w", "commanded_turn_angle_rad", "actual_accumulated_yaw_rad"
            )
        ),
        "turn_evaluation_fields_declared": all(
            value in evaluation_source for value in (
                "final_turn_angle_error_rad", "turn_completion_time_s", "straight_recovery_success",
                "post_turn_heading_error_rad", "turn_angle_results"
            )
        ),
        "crouch_relative_command_declared": all(
            value in command_source for value in (
                "crouch_entry_height", "crouch_commanded_drop", "crouch_return_progress"
            )
        ),
        "crouch_rewards_declared": all(
            value in reward_source for value in (
                "crouch_height_tracking", "CrouchDepthProgressReward", "crouch_return_height",
                "crouch_joint_symmetry", "crouch_foot_contact_loss", "crouch_joint_limit_proximity",
            )
        ),
    }
    print(json.dumps(result, indent=2))
    if not all(value for key, value in result.items() if isinstance(value, bool)):
        raise SystemExit("static smoke failed")


if __name__ == "__main__":
    main()
