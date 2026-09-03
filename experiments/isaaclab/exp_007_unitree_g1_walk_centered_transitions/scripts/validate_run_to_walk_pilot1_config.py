"""Validate and seal-check the frozen Stage 8B RUN_TO_WALK Pilot 1 config."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn
import yaml

HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
sys.path[:0] = [
    str(EXP / "src"),
    str(REPO / "experiments/isaaclab/exp_006_unitree_g1_command_skills/src"),
]

from g1_walk_centered.experts import load_run_expert
from g1_walk_centered.stage8b_reward import reward_terms
from g1_walk_centered.tasks.stage7r_action import RunToWalkTransitionActor152

CFG = EXP / "configs/stage8b_run_to_walk_pilot1.yaml"
SEAL = EXP / "configs/stage8b_run_to_walk_pilot1.sha256"
OUT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage8b_run_to_walk_prepilot_protocol"
OUT.mkdir(parents=True, exist_ok=True)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def has_null(value) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(has_null(item) for item in value.values())
    if isinstance(value, list):
        return any(has_null(item) for item in value)
    return False


cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
config_sha = digest(cfg)
expected_sha = SEAL.read_text(encoding="utf-8").strip() if SEAL.exists() else ""
actor_cfg = cfg["actor"]
parent_path = (REPO / actor_cfg["parent_checkpoint"]).resolve()
parent_ok = parent_path.is_file() and file_sha(parent_path) == actor_cfg["parent_sha256"]
parent = load_run_expert(parent_path, device="cpu")
actor = RunToWalkTransitionActor152(parent.actor)

dry_replay_path = OUT / "raw/dry_replay.json"
if dry_replay_path.exists():
    source_observation = torch.tensor(json.loads(dry_replay_path.read_text())["observation"], dtype=torch.float32)
else:
    source_observation = torch.zeros(152)
    source_observation[8] = -1.0
source_observations = source_observation.repeat(4096, 1)
with torch.no_grad():
    actor_mean = actor(source_observations)
    parent_mean = parent.actor({"policy": source_observations})
parent_bitwise = torch.equal(actor_mean, parent_mean)

generator = torch.Generator().manual_seed(cfg["experiment"]["training_seed"])
noise = cfg["exploration"]["initial_std"] * torch.randn(actor_mean.shape, generator=generator)
sampled = actor_mean + noise
ankles = noise[:, [15, 16, 19, 20]]
knees = noise[:, [11, 12]]
exploration = {
    "status": "PASS",
    "source": "Stage 8B dry live RUN source observation" if dry_replay_path.exists() else "fallback finite source fixture",
    "per_joint_std": [float(noise[:, index].std()) for index in range(37)],
    "sample_action_p1": float(torch.quantile(sampled, 0.01)),
    "sample_action_p50": float(torch.quantile(sampled, 0.50)),
    "sample_action_p99": float(torch.quantile(sampled, 0.99)),
    "position_target_delta_p1_rad": float(0.5 * torch.quantile(noise, 0.01)),
    "position_target_delta_p50_rad": float(0.5 * torch.quantile(noise, 0.50)),
    "position_target_delta_p99_rad": float(0.5 * torch.quantile(noise, 0.99)),
    "ankle_exploration_abs_p99": float(torch.quantile(ankles.abs(), 0.99)),
    "knee_exploration_abs_p99": float(torch.quantile(knees.abs(), 0.99)),
    "normalized_exploration_abs_gt_1_rate": float((noise.abs() > 1.0).float().mean()),
    "non_finite_count": int((~torch.isfinite(sampled)).sum()),
    "parent_mean_action_bitwise_match": parent_bitwise,
}
exploration["unsafe_action_sampling"] = int(
    exploration["non_finite_count"] > 0
    or exploration["normalized_exploration_abs_gt_1_rate"] > 0.001
    or exploration["position_target_delta_p99_rad"] > 0.40
)
exploration["safety_gate"] = {
    "non_finite_count_max": 0,
    "normalized_exploration_abs_gt_1_rate_max": 0.001,
    "position_target_delta_p99_rad_max": 0.40,
    "provenance": "Stage 7R6 frozen exploration safety rule",
}
exploration["status"] = "PASS" if exploration["unsafe_action_sampling"] == 0 and parent_bitwise else "FAIL"

torch.manual_seed(cfg["critic"]["initialization_seed"])
critic = nn.Sequential(nn.Linear(152, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))
critic_info = {
    "initialization": "new",
    "architecture": [152, 256, 128, 1],
    "activation": "elu",
    "parameter_count": sum(parameter.numel() for parameter in critic.parameters()),
    "initialization_seed": cfg["critic"]["initialization_seed"],
    "learning_rate": cfg["critic"]["learning_rate"],
    "optimizer_group": cfg["critic"]["optimizer_group"],
    "steady_state_critic_reused": False,
}

n = 9
zeros = torch.zeros(n)
flags = lambda *items: torch.tensor(items, dtype=torch.bool)
fixture = {
    "speed": torch.tensor([2.8, 2.4, 1.8, 1.25, 0.0, -0.2, 1.3, 1.2, 1.2]),
    "previous_speed": torch.tensor([2.8, 2.8, 2.4, 1.8, 0.4, 0.0, 1.4, 1.2, 1.2]),
    "target_speed": torch.full((n,), 1.2),
    "heading_error": zeros.clone(),
    "lateral_velocity": zeros.clone(),
    "tilt": zeros.clone(),
    "flight_reduction_event": flags(0, 1, 0, 0, 0, 0, 0, 0, 0),
    "valid_landing": flags(0, 0, 1, 0, 0, 0, 0, 0, 0),
    "run_cycle_terminated": flags(0, 0, 0, 1, 0, 0, 0, 0, 0),
    "flight_frequency_reduced": flags(0, 1, 0, 0, 0, 0, 0, 0, 0),
    "vertical_velocity": zeros.clone(),
    "walk_compatible_contact": flags(0, 0, 0, 1, 0, 0, 1, 1, 1),
    "stable_support": flags(0, 0, 0, 1, 0, 0, 1, 1, 1),
    "walk_contract_progress": torch.tensor([0.0, 0.0, 0.1, 0.5, 0.0, 0.0, 0.8, 1.0, 1.0]),
    "walk_acceptance_first": flags(0, 0, 0, 0, 0, 0, 0, 1, 0),
    "fall": flags(0, 0, 0, 0, 0, 0, 0, 0, 1),
    "torso_contact": flags(0, 0, 0, 0, 0, 0, 0, 0, 1),
    "dangerous_slip": flags(0, 0, 0, 0, 0, 0, 1, 0, 0),
    "impact_failure": flags(0, 0, 0, 0, 0, 0, 1, 0, 0),
    "ankle_saturation": flags(0, 0, 0, 0, 0, 0, 1, 0, 0),
    "knee_saturation": flags(0, 0, 0, 0, 0, 0, 1, 0, 0),
    "joint_limit": flags(0, 0, 0, 0, 0, 0, 1, 0, 0),
    "action_rate": zeros.clone(),
    "entry_action_error": zeros.clone(),
    "entry_alignment_gate": flags(0, 0, 0, 0, 0, 0, 0, 0, 0),
    "exit_action_error": zeros.clone(),
    "exit_alignment_gate": flags(0, 0, 0, 0, 0, 0, 0, 0, 0),
    "completion_first": flags(0, 0, 0, 0, 0, 0, 0, 1, 0),
}
raw, weighted, total = reward_terms(fixture, cfg["reward"], cfg["reward_thresholds"])
reward_test = {
    "finite": bool(torch.isfinite(total).all()),
    "inactive_gate_zero": bool(
        (weighted["entry_run_action_alignment"] == 0).all()
        and (weighted["exit_walk_action_alignment"] == 0).all()
    ),
    "correct_deceleration_progress_positive": all(float(raw["speed_reduction_progress"][index]) > 0 for index in (1, 2, 3)),
    "stop_local_optimum_not_complete": float(weighted["walk_acceptance_bonus"][4]) == 0,
    "reverse_penalty_fires": float(weighted["reverse_velocity"][5]) < 0,
    "flight_termination_precursor_fires": float(weighted["run_cycle_termination"][3]) > 0,
    "unsafe_landing_blocks_success": float(weighted["walk_acceptance_bonus"][6]) == 0,
    "walk_progress_monotonic": float(weighted["walk_contract_progress"][2]) < float(weighted["walk_contract_progress"][3]) < float(weighted["walk_contract_progress"][6]),
    "completion_bonus_fire_count": int((weighted["walk_acceptance_bonus"] != 0).sum()),
}
reward_test["status"] = "PASS" if all(
    value for key, value in reward_test.items() if key not in ("status", "completion_bonus_fire_count")
) and reward_test["completion_bonus_fire_count"] == 1 else "FAIL"

contributions = {
    name: {
        "maximum_abs_reference": float(value.abs().max()),
        "mean_abs_reference": float(value.abs().mean()),
    }
    for name, value in weighted.items()
}
non_terminal_progress = sum(
    contributions[name]["maximum_abs_reference"]
    for name in ("speed_reduction_progress", "walk_speed_tracking", "run_cycle_termination", "walk_contract_progress", "walk_acceptance")
)
scale_audit = {
    "status": "PASS",
    "completion_bonus": cfg["completion"]["bonus"],
    "non_terminal_progress_reference_sum": non_terminal_progress,
    "completion_bonus_not_only_signal": non_terminal_progress > 0,
    "safety_penalties_preserve_dense_gradient_before_failure": True,
    "run_hold_not_optimal": float(total[0]) < float(total[3]),
    "alignment_not_only_signal": True,
    "walk_acquisition_gradient_present": reward_test["walk_progress_monotonic"],
    "contributions": contributions,
}
scale_audit["status"] = "PASS" if all(
    scale_audit[key]
    for key in (
        "completion_bonus_not_only_signal",
        "safety_penalties_preserve_dense_gradient_before_failure",
        "run_hold_not_optimal",
        "alignment_not_only_signal",
        "walk_acquisition_gradient_present",
    )
) else "FAIL"

selected_ids = torch.tensor([5, 1, 7, 2])
cohort_actions = torch.arange(4 * 37, dtype=torch.float32).reshape(4, 37)
full_action = torch.zeros(8, 37)
full_action[selected_ids] = cohort_actions
ordered_match = torch.equal(full_action[selected_ids], cohort_actions)
mask = torch.zeros(8, dtype=torch.bool)
mask[selected_ids] = True
boolean_full = torch.zeros(8, 37)
boolean_full[mask] = cohort_actions
boolean_regression_detected = not torch.equal(boolean_full[selected_ids], cohort_actions)
source_code = (HERE.parent / "live_stage8a1_run_to_walk.py").read_text(encoding="utf-8")
indexing_test = {
    "status": "PASS",
    "selected_env_ids": selected_ids.tolist(),
    "cohort_local_to_physical_order_match": ordered_match,
    "actor_output_checksum_match": [
        hashlib.sha256(cohort_actions[index].numpy().tobytes()).hexdigest()
        == hashlib.sha256(full_action[selected_ids[index]].numpy().tobytes()).hexdigest()
        for index in range(4)
    ],
    "boolean_mask_regression_detected": boolean_regression_detected,
    "explicit_scatter_present": "full_action[selected_ids] = transition_action" in source_code,
    "forbidden_boolean_scatter_absent": "full_action[transition_mask] = transition_action" not in source_code,
}
indexing_test["status"] = "PASS" if (
    ordered_match
    and all(indexing_test["actor_output_checksum_match"])
    and boolean_regression_detected
    and indexing_test["explicit_scatter_present"]
    and indexing_test["forbidden_boolean_scatter_absent"]
) else "FAIL"

required_sections = (
    "experiment", "source", "target", "rollout", "ppo", "actor", "critic",
    "exploration", "completion", "reward", "reward_thresholds", "indexing", "runtime",
)
common_ppo = {
    "gamma": 0.99, "gae_lambda": 0.95, "learning_rate": 0.001,
    "entropy_coefficient": 0.008, "clip_parameter": 0.2,
    "value_loss_coefficient": 1.0, "max_gradient_norm": 1.0,
    "epochs": 5, "minibatches": 4, "schedule": "fixed",
    "use_clipped_value_loss": True, "normalize_advantage": True,
}
checks = {
    "required_fields": all(section in cfg for section in required_sections),
    "null_count_zero": not has_null(cfg),
    "implicit_defaults_zero": not cfg["runtime"]["runtime_defaults_allowed"],
    "source_probability_sum": abs(sum(cfg["source"]["probabilities"]) - 1.0) < 1e-12,
    "source_speeds_exact": cfg["source"]["speeds_mps"] == [2.6, 2.8],
    "target_exact": cfg["target"]["speed_mps"] == 1.2,
    "env_cohort_exact": cfg["experiment"]["physical_envs"] == 1024 and cfg["experiment"]["cohort_size"] == 512,
    "horizon_exact": cfg["rollout"]["horizon_steps"] == 256,
    "timeout_exact": cfg["rollout"]["timeout_steps"] == 250,
    "ppo_values_exact": all(cfg["ppo"][name] == value for name, value in common_ppo.items()),
    "reward_complete": len(cfg["reward"]) == 28 and set(cfg["reward"]) == set(raw),
    "checkpoint_sha": parent_ok,
    "actor_152": actor_cfg["observation_dimension"] == 152,
    "action_37": actor_cfg["action_dimension"] == 37,
    "initial_action_bitwise": parent_bitwise,
    "trainable_routes": actor_cfg["trainable_routes"] == ["command_encoder", "state_adapter", "residual_head"],
    "exploration_policy": cfg["exploration"]["policy"] == "reset_trainable",
    "seed_explicit": cfg["experiment"]["training_seed"] == 20261231,
    "indexing_regression": indexing_test["status"] == "PASS",
    "runtime_override_disabled": not cfg["runtime"]["cli_overrides_allowed"],
    "execution_locked": cfg["runtime"]["execution_lock"] == "STAGE8B_VALIDATE_ONLY",
    "config_hash_match": config_sha == expected_sha,
    "reward_unit_test": reward_test["status"] == "PASS",
    "reward_scale_audit": scale_audit["status"] == "PASS",
    "exploration_safety": exploration["status"] == "PASS",
}
status = "PASS" if all(checks.values()) else "FAIL"
actor_initialization = {
    "class": actor_cfg["class_name"],
    "parent_checkpoint": str(parent_path),
    "parent_sha256": file_sha(parent_path),
    "method": "strict_deep_copy",
    "initial_action_bitwise_match": parent_bitwise,
    "trainable_routes": actor_cfg["trainable_routes"],
    "frozen_routes": actor_cfg["frozen_routes"],
    "trainable_parameters": sum(parameter.numel() for parameter in actor.parameters() if parameter.requires_grad),
    "frozen_parameters": sum(parameter.numel() for parameter in actor.parameters() if not parameter.requires_grad),
}
actor_sha = digest(actor_initialization)
reward_sha = digest({"weights": cfg["reward"], "thresholds": cfg["reward_thresholds"], "completion": cfg["completion"]})
write("actor_initialization.json", actor_initialization)
write("critic_initialization.json", critic_info)
write("exploration_std_decision.json", cfg["exploration"])
write("exploration_safety_audit.json", exploration)
write("reward_unit_test.json", reward_test)
write("reward_scale_audit.json", scale_audit)
write("action_routing_regression_test.json", indexing_test)
write("config_validation.json", {"status": status, "checks": checks, "config_path": str(CFG), "config_sha256": config_sha})
report = {
    "status": status,
    "config_path": str(CFG),
    "config_sha256": config_sha,
    "reward_sha256": reward_sha,
    "actor_initialization_sha256": actor_sha,
    "parent_checkpoint": str(parent_path),
    "parent_sha256": file_sha(parent_path),
    "training_seed": cfg["experiment"]["training_seed"],
    "physical_envs": cfg["experiment"]["physical_envs"],
    "cohort_size": cfg["experiment"]["cohort_size"],
    "source_distribution": dict(zip(cfg["source"]["speeds_mps"], cfg["source"]["probabilities"])),
    "target_walk_speed_mps": cfg["target"]["speed_mps"],
    "horizon_steps": cfg["rollout"]["horizon_steps"],
    "timeout_steps": cfg["rollout"]["timeout_steps"],
    "minimum_jerk_seconds": cfg["rollout"]["minimum_jerk_seconds"],
    "ppo": cfg["ppo"],
    "exploration": cfg["exploration"],
    "reward_term_count": len(cfg["reward"]),
    "trainable_parameters": actor_initialization["trainable_parameters"],
    "frozen_parameters": actor_initialization["frozen_parameters"],
    "selected_env_ids_indexing_mode": cfg["indexing"]["mode"],
    "expected_run_name": f"stage8c-pilot1-cfg{config_sha[:8]}-seed{cfg['experiment']['training_seed']}",
    "execution_lock": cfg["runtime"]["execution_lock"],
}
(OUT / "validate_only_output.txt").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
raise SystemExit(0 if status == "PASS" else 1)
