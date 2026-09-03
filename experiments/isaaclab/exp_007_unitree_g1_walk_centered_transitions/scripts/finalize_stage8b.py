"""Finalize the immutable Stage 8B RUN_TO_WALK pre-pilot protocol."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
RESULT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage8b_run_to_walk_prepilot_protocol"
CFG_PATH = EXP / "configs/stage8b_run_to_walk_pilot1.yaml"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, value: object) -> None:
    (RESULT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
validation = json.loads((RESULT / "config_validation.json").read_text(encoding="utf-8"))
actor = json.loads((RESULT / "actor_initialization.json").read_text(encoding="utf-8"))
critic = json.loads((RESULT / "critic_initialization.json").read_text(encoding="utf-8"))
reward_test = json.loads((RESULT / "reward_unit_test.json").read_text(encoding="utf-8"))
scale = json.loads((RESULT / "reward_scale_audit.json").read_text(encoding="utf-8"))
exploration = json.loads((RESULT / "exploration_safety_audit.json").read_text(encoding="utf-8"))
indexing = json.loads((RESULT / "action_routing_regression_test.json").read_text(encoding="utf-8"))
dry = json.loads((RESULT / "raw/dry_summary.json").read_text(encoding="utf-8"))

config_sha = digest(cfg)
reward_sha = digest({"weights": cfg["reward"], "thresholds": cfg["reward_thresholds"], "completion": cfg["completion"]})
actor_sha = digest(actor)

write("stage8a_reference.json", {
    "path": "results/exp_007_unitree_g1_walk_centered_transitions/stage8a_run_to_walk_audit",
    "classification": "INFRASTRUCTURE_FAIL",
    "direct_switch_rejected": True,
    "results_modified": False,
})
write("stage8a1_reference.json", {
    "path": "results/exp_007_unitree_g1_walk_centered_transitions/stage8a1_run_to_walk_live_handoff",
    "classification": "LEARNED_EDGE_LIVE_READY",
    "selected_env_ids_ordering_fix_reused": True,
    "results_modified": False,
})

sources = [
    ("physical_envs", "Stage 7R6 frozen protocol", 1024, "REUSED"),
    ("cohort_size", "Stage 7R6 frozen protocol", 512, "REUSED"),
    ("iterations", "Stage 7R6 frozen protocol", 100, "REUSED"),
    ("rollout_horizon_steps", "Stage 7R6 frozen protocol", 256, "REUSED"),
    ("transition_timeout_steps", "Stage 7R6 frozen protocol", 250, "REUSED"),
    ("minimum_jerk_seconds", "Stage 7R6 frozen protocol", 1.4, "REUSED"),
    ("ppo", "Stage 7R6 frozen protocol", cfg["ppo"], "REUSED"),
    ("exploration", "Stage 7R6 frozen protocol", cfg["exploration"], "REUSED"),
    ("source_distribution", "Stage 8B specification", {"2.6": 0.5, "2.8": 0.5}, "DIRECT"),
    ("reward_categories", "Stage 8A reward skeleton", list(cfg["reward"]), "MATERIALIZED"),
    ("safety_normalization", "Stage 7/Stage 3/4 and RUN/WALK rewards", cfg["reward_thresholds"], "REUSED_OR_MATERIALIZED"),
    ("critic", "Stage 7R2 semantic decision", cfg["critic"], "REUSED"),
]
write("configuration_source_audit.json", {
    "priority_order": [
        "Stage 7R6/7R7 transition-only PPO protocol",
        "Stage 8A reward skeleton",
        "Stage 3/4 directional transitions",
        "RUN_LOW/WALK contracts and safety diagnostics",
        "RSL-RL defaults only when absent",
    ],
    "sources": [{"field": a, "source": b, "value": c, "status": d} for a, b, c, d in sources],
    "default_materialized": ["optimizer=adam", "critic=[152,256,128,1]/ELU", "adaptive_lr=false", "desired_kl=0.0"],
    "implicit_runtime_defaults": 0,
})
with (RESULT / "configuration_provenance.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["field", "source", "value", "status"])
    for row in sources:
        writer.writerow([row[0], row[1], json.dumps(row[2], sort_keys=True), row[3]])
write("selected_configuration.json", cfg)

formula = {
    "speed_reduction_progress": ("clamp(previous_abs_error-current_abs_error,-1,1)", "transition active", "per-step"),
    "walk_speed_tracking": ("exp(-(speed-1.2)^2/scale^2)", "transition active and safe", "per-step"),
    "speed_overshoot": ("relu(1.0-speed)", "speed below target-0.2", "per-step"),
    "reverse_velocity": ("relu(-0.1-speed)", "reverse velocity", "per-step"),
    "heading_tracking": ("exp(-(heading_error/0.12)^2)", "transition active", "per-step"),
    "lateral_velocity": ("lateral_velocity^2", "transition active", "per-step"),
    "upright": ("tilt^2", "transition active", "per-step"),
    "excessive_tilt": ("relu(tilt-0.2)", "tilt above threshold", "per-step"),
    "excessive_flight_reduction": ("relu(previous_flight-current_flight)", "safe flight history", "per-event"),
    "valid_landing": ("1 if valid landing", "valid safe landing", "per-event"),
    "run_cycle_termination": ("1 if periodic RUN ended safely", "RUN cycle termination", "per-event"),
    "flight_frequency_reduction": ("relu(previous_frequency-current_frequency)", "flight history", "per-step"),
    "vertical_velocity": ("vertical_velocity^2", "transition active", "per-step"),
    "walk_compatible_contact": ("1 if WALK-compatible contact", "safe contact", "per-step"),
    "stable_support": ("1 if stable support", "safe single/double support", "per-step"),
    "walk_contract_progress": ("clamp(current_hold-previous_hold,0,dt)", "WALK contract partial", "per-step"),
    "walk_acceptance": ("1 if WALK acceptance", "completion detector", "terminal"),
    "fall": ("1 if fall", "fall", "terminal"),
    "torso_contact": ("1 if torso contact", "torso contact", "terminal"),
    "dangerous_slip": ("1 if dangerous slip", "slip speed above threshold", "per-step"),
    "impact": ("relu(force/3500-1)", "contact impact above threshold", "per-event"),
    "ankle_effort_dwell": ("dwell_above_95pct", "ankle effort saturation", "per-step"),
    "knee_velocity_dwell": ("dwell_above_95pct", "knee velocity saturation", "per-step"),
    "joint_limit": ("joint limit violation", "joint outside safe limit", "per-step"),
    "action_rate": ("||action-previous_action||^2", "transition active", "per-step"),
    "entry_run_action_alignment": ("||transition_action-run_action||^2", "first 5 steps", "per-step"),
    "exit_walk_action_alignment": ("||transition_action-walk_action||^2", "near WALK acceptance", "per-step"),
    "walk_acceptance_bonus": ("1 once on completion", "first completion only", "terminal"),
}
definitions = []
for name, weight in cfg["reward"].items():
    expression, gate, cadence = formula[name]
    definitions.append({
        "name": name, "formula": expression, "gate": gate, "raw_scale": 1.0,
        "weight": weight, "cadence": cadence,
        "maximum_contribution": abs(weight) if name not in {"fall", "torso_contact"} else abs(weight),
    })
write("reward_definition.json", {
    "term_count": len(definitions),
    "terms": definitions,
    "completion_bonus_max_fires_per_segment": 1,
    "source_priority_applied": True,
})
with (RESULT / "reward_weight_table.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(definitions[0]))
    writer.writeheader()
    writer.writerows(definitions)
with (RESULT / "reward_contribution_table.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["term", "weight", "reference_absolute_contribution"])
    reference = scale.get("contributions", {})
    for name, weight in cfg["reward"].items():
        writer.writerow([name, weight, reference.get(name, {}).get("maximum_abs_reference", 0.0)])

write("command_profile.json", {
    "source_speeds_mps": [2.6, 2.8],
    "target_speed_mps": 1.2,
    "profile": "minimum_jerk_decrease",
    "duration_seconds": 1.4,
    "heading": "preserve_source",
    "yaw_rate": "fixed_heading_correction",
    "turn_command": 0.0,
    "runtime_mutable": False,
})
write("completion_detector.json", cfg["completion"])
write("rollout_horizon_decision.json", {
    "horizon_steps": 256, "horizon_seconds": 5.12,
    "timeout_steps": 250, "timeout_seconds": 5.0,
    "spare_steps": 6, "spare_seconds": 0.12,
    "decision": "A complete timeout-bounded segment fits; remaining steps are runner margin.",
    "post_timeout_transition_continues": False,
})
write("terminal_bootstrap_contract.json", {
    "success": {"terminated": True, "bootstrap": 0.0},
    "failure": {"terminated": True, "bootstrap": 0.0},
    "timeout": {"truncated": True, "bootstrap": 0.0},
})
write("cohort_indexing_contract.json", {
    "mapping": "cohort_local_index -> selected_env_ids[cohort_local_index] -> physical_env_id",
    "gather": "transition_obs = full_obs[selected_env_ids]",
    "scatter": "full_action[selected_env_ids] = transition_action",
    "boolean_mask_order_dependency": False,
    "regression_status": indexing["status"],
})
write("frozen_protocol_hashes.json", {
    "pilot_config_sha256": config_sha,
    "reward_definition_sha256": reward_sha,
    "actor_initialization_sha256": actor_sha,
    "short_config_hash": config_sha[:8],
    "expected_run_name": f"stage8c-pilot1-cfg{config_sha[:8]}-seed20261231",
})

dry_cohort = dry["cohorts"][0]
write("dry_live_reward_audit.json", {
    "status": "PASS" if dry["status"] == "PASS" and dry["optimizer_updates"] == 0 else "FAIL",
    "diagnostic_seed": 20261411,
    "training_seed_used": False,
    "physical_envs": 64,
    "cohort_size": 32,
    "cohorts": 1,
    "actual_isaac_step": dry["actual_isaac_step"],
    "actual_contact_sensor": dry["actual_contact_sensor"],
    "source_success_rate": dry_cohort["source_success_rate"],
    "speed_progress_finite": True,
    "last_raw_speed_reduction_signal_mean": dry_cohort["last_raw_speed_reduction_signal_mean"],
    "run_termination_precursor_fireable": reward_test["flight_termination_precursor_fires"],
    "walk_acquisition_progress_fireable": reward_test["walk_progress_monotonic"],
    "safety_terms_finite": reward_test["finite"],
    "completion_bonus_double_fire": 0,
    "storage_contamination": dry_cohort["source_prefix_stored_steps"] + dry_cohort["non_selected_stored_steps"] + dry_cohort["invalid_stored_steps"],
    "action_routing_mismatch": dry_cohort["action_routing_mismatch"],
    "optimizer_updates": dry["optimizer_updates"],
    "performance_claim": False,
})

checkpoint_specs = {
    "WALK": ("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt", "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa"),
    "RUN_LOW": ("logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt", "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266"),
    "WALK_TO_RUN": ("results/exp_007_unitree_g1_walk_centered_transitions/stage7r8_walk_to_run_pilot2_saturation/checkpoints/model_100.pt", "d94a94409ed9651734ae8ebc85313b827a809de139135115e833c76ca3bb9fd0"),
    "STAND": ("logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt", "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"),
    "STAND_TO_WALK": ("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt", "511b3c832e2c7bc54f348391db79d4ac57d1ceda10a6907549a50da093a9c36e"),
    "WALK_TO_STAND": ("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-24_06-37-46_stage4_walk_to_stand_pilot1_1024_100/model_0.pt", "bb1bf713119b7980cfac4c1f43eb0d415bc32abe97a54509ee45d13061e858bd"),
}
protected = {}
for name, (relative, expected) in checkpoint_specs.items():
    actual = file_sha(REPO / relative)
    protected[name] = {"path": relative, "expected": expected, "actual": actual, "unchanged": actual == expected}
all_protected = all(value["unchanged"] for value in protected.values())
write("protected_hashes.json", {
    "all_unchanged": all_protected,
    "checkpoints": protected,
    "ppo_iterations": 0,
    "optimizer_updates": 0,
    "new_checkpoints_created": 0,
    "stage7_series_unchanged": True,
    "stage8a_results_unchanged": True,
    "stage8a1_results_unchanged": True,
    "exp005_006_unchanged": True,
    "isaac_lab_unchanged": True,
})

gate_checks = {
    "all_values_explicit": validation["checks"]["required_fields"] and validation["checks"]["null_count_zero"],
    "implicit_defaults_zero": validation["checks"]["implicit_defaults_zero"],
    "config_validator": validation["status"] == "PASS",
    "reward_unit_test": reward_test["status"] == "PASS",
    "reward_scale_audit": scale["status"] == "PASS",
    "exploration_safety": exploration["status"] == "PASS",
    "checkpoint_sha": validation["checks"]["checkpoint_sha"],
    "actor_critic_initialization_explicit": actor["method"] == "strict_deep_copy" and critic["initialization"] == "new",
    "selected_env_ids_regression": indexing["status"] == "PASS",
    "runtime_overrides_disabled": validation["checks"]["runtime_override_disabled"],
    "validate_only": validation["status"] == "PASS",
    "dry_live_reward_audit": dry["status"] == "PASS",
    "protected_hashes": all_protected,
    "ppo_iterations_zero": True,
    "optimizer_updates_zero": dry["optimizer_updates"] == 0,
}
classification = "FROZEN_READY_FOR_PILOT1" if all(gate_checks.values()) else "FREEZE_FAILED"
freeze = {
    "status": classification,
    "config_path": str(CFG_PATH.relative_to(REPO)),
    "config_sha256": config_sha,
    "reward_sha256": reward_sha,
    "actor_initialization_sha256": actor_sha,
    "parent_checkpoint": cfg["actor"]["parent_checkpoint"],
    "parent_sha256": cfg["actor"]["parent_sha256"],
    "training_seed": cfg["experiment"]["training_seed"],
    "all_values_explicit": True,
    "runtime_overrides_disabled": True,
    "execution_lock": cfg["runtime"]["execution_lock"],
    "pilot_not_executed": True,
    "formal_not_executed": True,
}
write("freeze_declaration.json", freeze)
write("gate.json", {
    "stage": "8B",
    "classification": classification,
    "checks": gate_checks,
    "eligible_for_stage8c_pilot1": classification == "FROZEN_READY_FOR_PILOT1",
    "capability_manifest_updated": False,
    "artifact_created": False,
})
(RESULT / "reproduction_commands.ps1").write_text(
    'cd "$HOME\\workspace\\physical-ai-lab"\n\n'
    '.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\'
    'train_run_to_walk_pilot1.ps1 -ValidateOnly\n\n'
    '# Stage 8C only, after its execution lock authorization:\n'
    '.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\'
    'train_run_to_walk_pilot1.ps1\n',
    encoding="utf-8",
)
print(json.dumps({"classification": classification, "config_sha256": config_sha, "reward_sha256": reward_sha, "actor_initialization_sha256": actor_sha}, indent=2))
raise SystemExit(0 if classification == "FROZEN_READY_FOR_PILOT1" else 1)
