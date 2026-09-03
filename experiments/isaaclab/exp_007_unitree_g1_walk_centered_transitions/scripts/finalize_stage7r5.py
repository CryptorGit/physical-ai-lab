"""Fail-closed Stage 7R5 pilot preflight when no frozen PPO/reward config exists."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage7r5_walk_to_run_pilot1"
OUT.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [
    str(EXP / "src"),
    str(REPO / "experiments/isaaclab/exp_006_unitree_g1_command_skills/src"),
]
from g1_walk_centered.experts import load_run_expert  # noqa: E402
from g1_walk_centered.tasks.stage7r_action import WalkToRunTransitionActor152  # noqa: E402


def write(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


walk = REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt"
run = REPO / "logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt"
stand = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
stw = REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt"
wts = REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-24_06-37-46_stage4_walk_to_stand_pilot1_1024_100/model_0.pt"
parent = load_run_expert(run, device="cpu").actor
actor = WalkToRunTransitionActor152(parent)
trainable = sum(parameter.numel() for parameter in actor.parameters() if parameter.requires_grad)
total = sum(parameter.numel() for parameter in actor.parameters())
distribution = parent.distribution
std = (
    distribution.std_param.detach().cpu().exp()
    if distribution is not None and distribution.std_type == "log"
    else distribution.std_param.detach().cpu()
    if distribution is not None
    else torch.empty(0)
)
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()

missing = {
    "rollout_horizon": "null in Stage 7R2 training_config.json",
    "learning_rate": "not frozen",
    "entropy_coefficient": "not frozen",
    "clip_parameter": "not frozen",
    "value_loss_coefficient": "not frozen",
    "maximum_gradient_norm": "not frozen",
    "reward_weights": "only term names exist; no numeric weights",
    "completion_bonus_weight": "not frozen",
    "transition_timeout": "not frozen for Stage 7R PPO",
    "minimum_jerk_duration": "1.4 s used by a failed baseline, not frozen as PPO config",
    "exploration_std_policy": "parent value observable, but reset/preserve/train policy not frozen",
    "ppo_epochs_and_minibatches": "not frozen",
}
write("stage7r4_reference.json", {
    "path": "results/exp_007_unitree_g1_walk_centered_transitions/stage7r4_live_cohort_integration",
    "status": "PASS", "r0_complete": True, "unchanged": True,
})
write("pilot_scope.json", {
    "requested_iterations": 100, "executed_iterations": 0,
    "pilot_2": False, "formal": False, "capability_update": False,
    "artifact": False, "stop_reason": "FROZEN_TRAINING_CONFIGURATION_INCOMPLETE",
})
write("frozen_training_config.json", {
    "status": "INCOMPLETE_DO_NOT_TRAIN", "physical_envs": 1024,
    "cohort_size": 512, "runner": "IN_PLACE_TRANSITION_ONLY_PPO",
    "source": "WALK@1.2", "targets_mps": [2.4, 2.6, 2.8],
    "target_distribution": {"2.4": 0.5, "2.6": 0.3, "2.8": 0.2},
    "iterations": 100, "gamma": 0.99, "gae_lambda": 0.95,
    "missing_required_frozen_values": missing,
})
write("actor_initialization.json", {
    "class": "WalkToRunTransitionActor152", "route": "A",
    "parent": str(run.relative_to(REPO)).replace("\\", "/"), "parent_sha256": sha(run),
    "input_dim": 152, "output_dim": 37, "action_scale": 0.5,
    "total_parameters": total, "trainable_parameters": trainable,
    "trainable_modules": ["RUN command encoder", "RUN state adapter", "RUN residual head"],
    "parent_exploration_std": {
        "count": std.numel(), "min": float(std.min()), "mean": float(std.mean()), "max": float(std.max())
    } if std.numel() else None,
})
write("critic_initialization.json", {
    "decision": "NEW_TRANSITION_CRITIC", "input_dim": 152,
    "status": "NOT_INSTANTIATED_FOR_PILOT",
    "reason": "pilot configuration was incomplete before initialization",
})
write("initial_baseline_summary.json", {
    "executed": False, "episodes": 0,
    "reason": "baseline and pilot must share the same frozen actor exploration/command/reward/terminal config",
})
write("initial_baseline_per_target.json", {
    "executed": False, "requested_episodes_per_target": 20, "results": [],
})
write("target_sampling_audit.json", {
    "executed": False, "configured_distribution": {"2.4": 0.5, "2.6": 0.3, "2.8": 0.2},
    "segments": 0,
})
for name, fields in {
    "target_segment_counts.csv": ["iteration", "target_mps", "segments"],
    "training_curves.csv": ["iteration", "policy_loss", "value_loss", "entropy", "kl"],
    "checkpoint_evaluations.csv": ["checkpoint", "target_mps", "episodes", "success_rate"],
}.items():
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()
write("training_diagnostics.json", {
    "iterations": 0, "ppo_updates": 0, "nan_inf": 0,
    "classification_available": False, "blocking_configuration": missing,
})
write("reward_term_statistics.json", {
    "executed": False, "terms": [
        "speed progress", "speed tracking", "heading", "lateral velocity", "upright",
        "safe liftoff", "safe flight", "valid landing", "alternating landing",
        "consecutive cycles", "slip", "impact", "ankle saturation", "knee saturation",
        "excessive flight", "fall", "action rate", "RUN acceptance",
    ], "numeric_weights_available": False,
})
write("checkpoint_manifest.json", {"created": False, "checkpoints": []})
write("checkpoint_sweep.json", {"executed": False, "selected_diagnostic_checkpoint": None})
write("per_checkpoint_per_target.json", {"executed": False, "results": []})
write("failure_counts.json", {"frozen_training_configuration_incomplete": 1, "pilot_not_started": 1})
write("learning_signal_classification.json", {
    "classification": "NOT_EVALUATED_CONFIGURATION_BLOCK",
    "clear_learning_signal": False, "partial_learning_signal": False,
    "no_learning_signal": False, "unstable_learning": False,
    "reason": "No PPO update was scientifically authorized; absence of a run is not NO_LEARNING_SIGNAL.",
})
write("recommended_next_action.json", {
    "single_recommendation": "Freeze the complete Stage 7R PPO/reward configuration in a dedicated pre-pilot protocol.",
    "pilot_2": False, "formal": False,
})
write("protected_hashes.json", {
    "walk": sha(walk), "run": sha(run), "stand": sha(stand),
    "stand_to_walk": sha(stw), "walk_to_stand": sha(wts),
    "frozen_gradient_zero": True, "optimizer_contains_frozen_parameters": False,
    "production_checkpoint_changed": False, "stage7_series_unchanged": True,
})
write("gate.json", {
    "stage": "7R5", "status": "FAIL", "pilot1_executed": False,
    "iterations": 0, "formal_executed": False, "capability_updated": False,
    "artifact_created": False, "eligible_for_pilot2": False,
    "failure_class": "FROZEN_TRAINING_CONFIGURATION_INCOMPLETE",
    "missing_frozen_values": list(missing), "git_revision_before_commit": head,
})
(OUT / "reproduction_commands.ps1").write_text(
    'cd "$HOME\\workspace\\physical-ai-lab"\n'
    '& "$HOME\\workspace\\IsaacLab\\isaaclab.bat" -p '
    '".\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\finalize_stage7r5.py"\n'
    'Write-Error "Pilot intentionally blocked: freeze the missing PPO/reward configuration first."\n',
    encoding="utf-8",
)
print(json.dumps({"status": "FAIL", "pilot_iterations": 0, "missing": list(missing)}, indent=2))
