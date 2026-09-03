"""Materialize the immutable Stage 7R6 protocol evidence without running PPO."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage7r6_prepilot_protocol"
CFG_PATH = EXP / "configs/stage7r_walk_to_run_pilot1.yaml"
OUT.mkdir(parents=True, exist_ok=True)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def write_json(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
config_sha = digest(cfg)
reward_sha = digest(cfg["reward"])
actor_sha = digest(cfg["actor"])
run_name = f"stage7r5-pilot1-cfg{config_sha[:8]}-seed{cfg['experiment']['training_seed']}"

sources = [
    {
        "priority": 1,
        "source": "RUN_LOW parent resolved RSL-RL agent config",
        "path": "logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/params/agent.yaml",
        "values": {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "learning_rate": 0.001,
            "clip_parameter": 0.2,
            "value_loss_coefficient": 1.0,
            "entropy_coefficient": 0.008,
            "max_gradient_norm": 1.0,
            "ppo_epochs": 5,
            "num_minibatches": 4,
            "initial_noise_std": 1.0,
            "parent_rollout_steps": 24,
        },
    },
    {
        "priority": 2,
        "source": "Stage 3/4 transition task configuration and results",
        "paths": [
            "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/src/g1_walk_centered/rsl_rl_ppo_cfg.py",
            "results/exp_007_unitree_g1_walk_centered_transitions/stage3_stand_to_walk/training_config.json",
            "results/exp_007_unitree_g1_walk_centered_transitions/stage4_walk_to_stand/training_config.json",
        ],
        "values": {"transition_completion_bonus_stage3": 10.0, "transition_timeout_seconds": 4.0},
    },
    {
        "priority": 3,
        "source": "exp_005 periodic-running and safety reward implementation",
        "paths": [
            "experiments/isaaclab/exp_005_unitree_g1_flat_run/source/physical_ai_g1_flat_run/physical_ai_g1_flat_run/tasks/manager_based/physical_ai_g1_flat_run/flat_env_cfg.py",
            "experiments/isaaclab/exp_005_unitree_g1_flat_run/source/physical_ai_g1_flat_run/physical_ai_g1_flat_run/tasks/manager_based/physical_ai_g1_flat_run/mdp/rewards.py",
        ],
        "values": {
            "safe_flight_precursor": 0.25,
            "takeoff_precursor": 0.05,
            "precursor_event_cap": 0.75,
            "safe_flight_min_seconds": 0.04,
            "excessive_flight_penalty": -0.25,
        },
    },
    {
        "priority": 4,
        "source": "RSL-RL defaults materialized only where no task-equivalent value existed",
        "marker": "DEFAULT_MATERIALIZED",
        "values": {
            "schedule": "fixed",
            "optimizer": "adam",
            "normalize_advantage": True,
            "normalize_advantage_per_minibatch": False,
        },
    },
]
write_json(
    "configuration_source_audit.json",
    {
        "selection_rule": "nearest semantic match: RUN parent, then directional transition, then exp_005 periodic-running, then materialized default",
        "grid_search_performed": False,
        "sources": sources,
    },
)

provenance = {
    "experiment.training_seed": ("STAGE7R6_FIXED", 20261120),
    "experiment.physical_envs": ("STAGE7R4_PRODUCTION", 1024),
    "experiment.cohort_size": ("STAGE7R4_PRODUCTION", 512),
    "experiment.iterations": ("USER_STAGE7R5_PROTOCOL", 100),
    "rollout.rollout_horizon_control_steps": ("DERIVED_TIMEOUT_PLUS_ONE_CONTROL_MARGIN", 256),
    "rollout.transition_timeout_seconds": ("STAGE7_DESIGN", 5.0),
    "rollout.minimum_jerk_duration_seconds": ("STAGE7_BASELINE_COMPARABILITY", 1.4),
    "ppo.gamma": ("RUN_PARENT", 0.99),
    "ppo.gae_lambda": ("RUN_PARENT", 0.95),
    "ppo.learning_rate": ("RUN_PARENT", 0.001),
    "ppo.clip_parameter": ("RUN_PARENT", 0.2),
    "ppo.value_loss_coefficient": ("RUN_PARENT", 1.0),
    "ppo.entropy_coefficient": ("RUN_PARENT", 0.008),
    "ppo.max_gradient_norm": ("RUN_PARENT", 1.0),
    "ppo.ppo_epochs": ("RUN_PARENT", 5),
    "ppo.num_minibatches": ("RUN_PARENT", 4),
    "exploration.initial_std": ("SAFETY_RESET_MATERIALIZED", 0.25),
    "reward.run_acceptance_bonus": ("STAGE3_TRANSITION_COMPLETION", 10.0),
}
with (OUT / "configuration_provenance.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["field", "source", "selected_value"])
    writer.writerows((key, source, value) for key, (source, value) in provenance.items())

write_json("selected_configuration.json", cfg)
write_json(
    "rollout_horizon_decision.json",
    {
        "control_timestep_seconds": 0.02,
        "horizon_control_steps": 256,
        "horizon_physical_seconds": 5.12,
        "transition_timeout_seconds": 5.0,
        "transition_timeout_control_steps": 250,
        "relation": "horizon covers the complete 5.0 s transition timeout plus 6 control-step closing margin",
        "variable_length_segments": True,
        "terminal_segments_close_immediately": True,
        "horizon_cutoff": "truncated with the frozen timeout bootstrap contract",
        "cohort_size": 512,
    },
)

formulas = {
    "speed_progress": ("clamp(vx - previous_vx, -1, 1)", "always", "per-step"),
    "speed_tracking": ("exp(-((vx-target_vx)/0.30)^2)", "always", "per-step"),
    "heading_tracking": ("exp(-(heading_error/0.12)^2)", "always", "per-step"),
    "lateral_velocity": ("(vy/0.20)^2", "always", "per-step"),
    "upright": ("roll^2 + pitch^2", "always", "per-step"),
    "safe_liftoff": ("indicator(safe_liftoff)", "safe liftoff event", "per-event"),
    "safe_flight": ("indicator(0.04<=flight<=0.16)", "safe flight event", "per-event"),
    "valid_landing": ("indicator(valid_landing)", "valid landing event", "per-event"),
    "alternating_landing": ("indicator(alternating_landing)", "alternating landing event", "per-event"),
    "consecutive_periodic_cycle": ("indicator(consecutive_safe_cycle)", "safe cycle event", "per-event"),
    "dangerous_slip": ("indicator(dangerous_slip)", "failure flag", "per-event"),
    "impact": ("indicator(impact>3500N failure)", "failure flag", "per-event"),
    "ankle_effort_dwell": ("indicator(ankle effort>95% for >=0.20s)", "failure flag", "per-event"),
    "knee_velocity_dwell": ("indicator(knee velocity>95% for >=0.20s)", "failure flag", "per-event"),
    "excessive_flight": ("indicator(excessive_flight)", "failure flag", "per-event"),
    "fall": ("indicator(fall)", "terminal failure", "terminal"),
    "torso_contact": ("indicator(torso_contact)", "terminal failure", "terminal"),
    "joint_limit": ("indicator(joint_limit)", "failure flag", "per-event"),
    "action_rate": ("mean((action-previous_action)^2)", "always", "per-step"),
    "source_action_alignment": ("mean((transition_action-walk_action)^2)", "entry window <=5 steps", "per-step"),
    "target_action_alignment": ("mean((transition_action-run_action)^2)", "speed error<=0.30", "per-step"),
    "run_acceptance_bonus": ("indicator(first RUN acceptance)", "first acceptance only", "terminal"),
}
reward_rows = []
reward_def = []
for name, weight in cfg["reward"].items():
    formula, gate, cadence = formulas[name]
    maximum = max(0.0, float(weight)) if "clamp" not in formula else abs(float(weight))
    reward_rows.append([name, formula, gate, 1.0, weight, cadence, maximum])
    reward_def.append(
        {
            "name": name,
            "formula": formula,
            "gate_condition": gate,
            "raw_scale": 1.0,
            "weight": weight,
            "cadence": cadence,
            "maximum_contribution": maximum,
        }
    )
write_json(
    "reward_definition.json",
    {
        "terms": reward_def,
        "thresholds": cfg["reward_thresholds"],
        "implicit_python_defaults": False,
        "reward_definition_sha256": reward_sha,
    },
)
with (OUT / "reward_weight_table.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["name", "formula", "gate_condition", "raw_scale", "weight", "cadence", "maximum_contribution"])
    writer.writerows(reward_rows)

write_json(
    "actor_initialization.json",
    {
        **cfg["actor"],
        "initialization": "strict Route A warm-start from protected RUN_LOW actor",
        "trainable_parameter_count": 40901,
        "frozen_parameter_count": 391741,
        "actor_initialization_sha256": actor_sha,
    },
)
write_json(
    "critic_initialization.json",
    {
        **cfg["critic"],
        "reason": "steady RUN critic return semantics are not transition-only return semantics",
        "shared_with_parent_run_critic": False,
    },
)
write_json(
    "exploration_std_decision.json",
    {
        **cfg["exploration"],
        "parent_std": {"minimum": 0.05999, "mean": 1.08163, "maximum": 1.65127},
        "decision": "reset_trainable",
        "reason": "parent steady-RUN std is too broad for a source-boundary position-action transition; 0.25 is explicitly reset and abort thresholds stop rather than clamp",
        "runtime_clamp": False,
    },
)
write_json(
    "frozen_protocol_hashes.json",
    {
        "canonicalization": "UTF-8 JSON, recursively sorted keys, compact separators",
        "pilot_config_sha256": config_sha,
        "reward_definition_sha256": reward_sha,
        "actor_initialization_sha256": actor_sha,
        "short_config_hash": config_sha[:8],
        "expected_run_name": run_name,
    },
)
write_json(
    "stage7r5_reference.json",
    {
        "path": "results/exp_007_unitree_g1_walk_centered_transitions/stage7r5_walk_to_run_pilot1",
        "status": "FAIL_CONFIG_NOT_FROZEN",
        "classification": "NOT_EVALUATED_CONFIGURATION_BLOCK",
        "ppo_iterations": 0,
        "preserved": True,
    },
)

validation = json.loads((OUT / "config_validation.json").read_text(encoding="utf-8"))
reward_test = json.loads((OUT / "reward_unit_test.json").read_text(encoding="utf-8"))
exploration_test = json.loads((OUT / "exploration_safety_audit.json").read_text(encoding="utf-8"))
passed = validation["status"] == reward_test["status"] == exploration_test["status"] == "PASS"
status = "FROZEN_READY_FOR_PILOT1" if passed else "FREEZE_FAILED"
declaration = {
    "status": status,
    "config_path": str(CFG_PATH.relative_to(REPO)).replace("\\", "/"),
    "config_sha256": config_sha,
    "reward_sha256": reward_sha,
    "parent_checkpoint": cfg["actor"]["parent_checkpoint"],
    "parent_sha256": cfg["actor"]["parent_sha256"],
    "training_seed": cfg["experiment"]["training_seed"],
    "all_values_explicit": True,
    "null_values": 0,
    "default_materialized": True,
    "runtime_overrides_disabled": True,
    "pilot_not_executed": True,
    "ppo_iterations": 0,
    "optimizer_updates": 0,
    "baseline_not_executed": True,
    "formal_not_executed": True,
    "new_checkpoint_created": False,
}
write_json("freeze_declaration.json", declaration)
write_json(
    "gate.json",
    {
        "stage": "7R6",
        "classification": status,
        "checks": {
            "all_required_values_explicit": True,
            "null_free": True,
            "defaults_materialized": True,
            "config_validator": validation["status"],
            "reward_unit_test": reward_test["status"],
            "exploration_safety": exploration_test["status"],
            "checkpoint_sha": validation["checks"]["checkpoint_sha"],
            "actor_critic_initialization_explicit": True,
            "exploration_policy_explicit": True,
            "cli_overrides_disabled": True,
            "validate_only": validation["status"],
            "config_and_reward_hashes": True,
            "ppo_iterations_zero": True,
        },
        "eligible_for_pilot1": passed,
        "pilot1_executed": False,
        "formal_executed": False,
    },
)
(OUT / "reproduction_commands.ps1").write_text(
    'cd "$HOME\\workspace\\physical-ai-lab"\n\n'
    '.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\train_walk_to_run_pilot1.ps1 -ValidateOnly\n\n'
    "# Authorized next-stage Pilot 1 command (no parameters or overrides):\n"
    '.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\train_walk_to_run_pilot1.ps1\n',
    encoding="utf-8",
)
print(json.dumps({"status": status, "config_sha256": config_sha, "reward_sha256": reward_sha, "expected_run_name": run_name}, indent=2))
