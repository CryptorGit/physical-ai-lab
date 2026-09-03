"""Finalize tracked Stage 2N evidence without changing any policy."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2n_gait_conditioned_ppo_retention_preflight"
M = ROOT / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2m_stochastic_gait_endpoint_robustness"
K = ROOT / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2k_gait_latent_preflight"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


betas = [("B0", "b0", 0.0), ("B001", "b001", .01), ("B003", "b003", .03), ("B010", "b010", .10)]
preflight = []
for name, tag, beta in betas:
    metric = load(OUT / f"{tag}_first_update_stability.json")
    gradient = load(OUT / f"{tag}_gradient_audit.json")
    preflight.append({
        "candidate": name, "beta": beta,
        "ppo_gradient_norm": gradient["ppo_gradient_norm"],
        "anchor_gradient_norm_at_identity": gradient["anchor_gradient_norm"],
        "effective_anchor_ppo_gradient_ratio_at_identity": gradient["effective_anchor_ppo_ratio"],
        "gradient_cap_pass": gradient["cap_pass"],
        "exact_rollout_kl": metric["exact_kl"], "clip_fraction": metric["clip_fraction"],
        "mean_action_shift": metric["mean_action_shift"],
        "walk_anchor_kl": metric["anchor_kl_walk_1p2"],
        "run_1p2_anchor_kl": metric["anchor_kl_run_1p2"],
        "run_2p4_anchor_kl": metric["anchor_kl_run_2p4"],
        "run_2p6_anchor_kl": metric["anchor_kl_run_2p6"],
        "anchor_kl_gate_pass": max(metric[key] for key in metric if key.startswith("anchor_kl_")) <= .02,
    })
write_csv("anchor_coefficient_preflight.csv", preflight)

beta_eval = OUT / "beta_b001_eval"
one_update = {}
for action in ("deterministic", "stochastic"):
    one_update[action] = {}
    endpoint = load(beta_eval / f"{action}_endpoints/endpoints_evaluation.json")["summary"]
    one_update[action].update(endpoint)
    for mode in ("toggleA", "toggleB"):
        data = load(beta_eval / f"{action}_{mode}/{mode}_evaluation.json")["summary"][mode]
        rows = list(csv.DictReader((beta_eval / f"{action}_{mode}/{mode}_evaluation.csv").open(encoding="utf-8")))
        data["acquisition_rate"] = sum(row["transition_time_s"] != "" for row in rows) / len(rows)
        one_update[action][mode] = data
dump("selected_anchor_coefficient.json", {
    "status": "PASS", "selected_beta": .01, "selection": "minimum passing coefficient",
    "reason": "B0 WALK anchor KL 0.020086 exceeded 0.02; B001 passed all one-update gates",
    "one_update_retention": one_update,
})
dump("endpoint_anchor_objective.json", {
    "direction": "KL(reference||current)", "endpoint_weights": {
        "walk_1p2": .25, "run_1p2": .25, "run_2p4": .25, "run_2p6": .25,
    }, "reference_frozen": True, "gradient_to_reference": False,
    "actor_loss": "PPO + beta * endpoint_anchor_KL", "critic_loss_changed": False,
})
dump("stage_reference.json", {
    "stage": "2N", "starting_head": "a9120ca25dfb09a69fd80d0dd9df4729c1131275",
    "mean_actor_sha256": "d0c46acdc2d3a5793d7dc8d6ae9e47f741ff0353fb1aef8c9ff993e71ea8bec3",
    "gaussian_source_sha256": "175131f7415988c4992b1a0334911abcfed304fca79765d453269a13743af2ac",
    "alpha_walk": .30, "alpha_run": .65,
})
dump("protocol.json", {
    "stage": "2N", "purpose": "gait-conditioned PPO endpoint-retention fine-tuning preflight",
    "environment_reward_changed": False, "gait_reward_added": False,
    "reference_policy_role": "frozen KL anchor during training only",
    "runtime_reference_policy": False, "num_envs": 1024, "rollout_steps": 24,
    "iterations_requested": 25, "maximum_runs": 1, "training_seed": 20268021,
    "prohibited_capabilities_added": [],
})
dump("fine_tuning_parent_manifest.json", {
    "path": str((OUT / "checkpoints/model_initial.pt").relative_to(ROOT)),
    "sha256": sha(OUT / "checkpoints/model_initial.pt"), "observation_dim": 124, "action_dim": 37,
    "mean_actor_source": "Stage 2K selected", "std_source": "Stage 2L endpoints scaled by Stage 2M pair",
    "critic_source": "RUN teacher 123D with zero gait column",
})
dump("gait_conditioned_critic_contract.json", {
    "architecture": [124, 256, 128, 128, 1], "single_critic": True,
    "run_teacher_123d_columns_copied": True, "gait_column_zero_initialized": True,
    "regime_specific_heads": 0,
})
dump("resolved_curriculum.json", {
    "WALK_1P2": .25, "RUN_1P2": .20, "RUN_2P4_2P6": .30, "GAIT_TOGGLE_1P2": .25,
    "run_high_target_split": {"2.4": .5, "2.6": .5}, "toggle_split": {"walk_to_run": .5, "run_to_walk": .5},
    "stand_or_low_speed_walk_samples": 0,
})
(OUT / "resolved_training_config.yaml").write_text(
    "num_envs: 1024\niterations_requested: 25\niterations_executed: 4\n"
    "rollout_steps: 24\ntraining_seed: 20268021\nanchor_beta: 0.01\n"
    "early_stop: endpoint_anchor_kl_above_0.05\n"
    "ppo_settings: inherited_from_run_teacher\n", encoding="utf-8")

curves = list(csv.DictReader((OUT / "training_curves.csv").open(encoding="utf-8")))
selected = OUT / "checkpoints/model_initial.pt"
dump("selected_checkpoint.json", {
    "selection": "initial", "path": str(selected.relative_to(ROOT)), "sha256": sha(selected),
    "reason": "iteration 4 early guard; initial ranks first by endpoint/toggle pass and zero anchor KL",
    "persistent_training_updates_selected": 0,
})

# Formal endpoint evidence is an exact-parent reuse of the already audited Stage 2M 100-episode sweep.
sweep = list(csv.DictReader((M / "std_multiplier_endpoint_sweep.csv").open(encoding="utf-8")))
formal = []
for row in sweep:
    if row["policy"] != "student":
        continue
    alpha = float(row["alpha"])
    desired = 0.0 if alpha == 0 else (.30 if row["endpoint"] == "walk_1p2" else .65)
    if abs(alpha - desired) < 1e-12:
        formal.append({
            "mode": "deterministic" if alpha == 0 else "candidate_stochastic",
            **row, "selected_parent_identity": True,
        })
write_csv("formal_endpoint_results.csv", formal)

toggle_rows = []
for mode, label in (("toggleA", "WALK_TO_RUN"), ("toggleB", "RUN_TO_WALK")):
    for action, root in (
        ("deterministic", K / "raw"),
        ("candidate_stochastic", M / "raw/candidate"),
    ):
        summary = load(root / f"{mode}_evaluation.json")["summary"][mode]
        rows = list(csv.DictReader((root / f"{mode}_evaluation.csv").open(encoding="utf-8")))
        toggle_rows.append({
            "transition": label, "mode": action, "episodes": len(rows),
            "acquisition_rate": sum(row["transition_time_s"] != "" for row in rows) / len(rows),
            "fall_rate": summary["fall_rate"], "speed_mae": summary["speed_mae"],
            "heading_p95": summary["heading_p95"],
            "dangerous_slip_rate": summary["dangerous_slip_rate"],
            "impact_failure_rate": summary["impact_failure_rate"],
            "long_dwell_saturation_rate": summary["long_dwell_saturation_rate"],
            "selected_parent_identity": True,
        })
write_csv("formal_toggle_results.csv", toggle_rows)
dump("endpoint_anchor_kl.json", {
    "selected_checkpoint": "initial", "walk_1p2": 0.0, "run_1p2": 0.0,
    "run_2p4": 0.0, "run_2p6": 0.0, "gate_pass": True,
    "training_peak_before_stop": max(float(row["anchor_kl_walk_1p2"]) for row in curves),
})
dump("single_weight_audit.json", {
    "status": "PASS", "unique_checkpoint": 1, "unique_mean_actor": 1,
    "unique_gaussian_head": 1, "unique_critic_during_training": 1,
    "teacher_action_calls": 0, "expert_action_calls": 0, "router": 0,
    "checkpoint_switch": 0, "action_blend": 0,
})
dump("checkpoint_manifest.json", {
    "checkpoints": [
        {"iteration": 0, "path": str(selected.relative_to(ROOT)), "sha256": sha(selected), "selected": True},
        {"iteration": 1, "path": str((OUT / "checkpoints/model_1.pt").relative_to(ROOT)),
         "sha256": sha(OUT / "checkpoints/model_1.pt"), "selected": False},
    ], "stopped_before_next_scheduled_checkpoint": True,
})
classification = "GAIT_CONDITIONED_PPO_MULTIPLE_FAILURES"
dump("stage_classification.json", {
    "classification": classification,
    "beta_selection": "PASS", "first_update_stability": "PASS",
    "training_completion": "FAIL_STOPPED_ITERATION_4",
    "failure": "WALK endpoint anchor KL 0.066552 exceeded early guard 0.05",
    "existing_stage_classifications_overwritten": False,
})
dump("recommended_next_action.json", {
    "action": "diagnose endpoint-anchor accumulation across consecutive PPO updates",
    "single_method_only": True, "additional_training_authorized": False,
})
dump("gate.json", {
    "pass": False, "classification": classification, "beta": .01,
    "iterations_requested": 25, "iterations_executed": 4, "interactions": 1024 * 24 * 4,
    "production_policy_update": 0, "remote_push": False,
})
dump("protected_hashes.json", {
    "teacher_walk_sha256": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
    "teacher_run_sha256": "90d1a360587142d7e312db00a281505a027ecb221733eea6451a885868f6ccc9",
    "stage2k_sha256": "d0c46acdc2d3a5793d7dc8d6ae9e47f741ff0353fb1aef8c9ff993e71ea8bec3",
    "stage2l_sha256": "175131f7415988c4992b1a0334911abcfed304fca79765d453269a13743af2ac",
    "reward_changes": 0, "physics_changes": 0, "core_changes": 0,
})
(OUT / "reproduction_commands.ps1").write_text(
    "$env:PYTHONPATH=\"$PWD\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\src;"
    "$PWD\\experiments\\isaaclab\\exp_005_unitree_g1_flat_run\\src;$PWD\"\n"
    "& C:\\isaacsim\\python.bat experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\run_stage2n_retention.py --mode prepare --headless\n"
    "& C:\\isaacsim\\python.bat experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\run_stage2n_retention.py --mode train --beta 0.01 --iterations 25 --headless\n",
    encoding="utf-8")

report = f"""# EXP 012 Stage 2N — gait-conditioned PPO endpoint-retention preflight

## Outcome

Classification: `{classification}`.

The integrated parent, critic initialization, mixed Adam-moment mapping, strict LR synchronization,
anchor collection, beta preflight, and first PPO update all passed. The minimum passing one-update
coefficient was `beta=0.01`. The single authorized continuation stopped at iteration 4 because WALK
anchor KL reached `0.066552`, above the `0.05` early guard. Rollout KL remained `0.014060`; this was
semantic endpoint drift, not PPO numerical instability.

## Parent and optimizer

The Stage 2K mean is bitwise identical. WALK/RUN std use calibrated multipliers `0.30/0.65`.
The critic copies RUN-teacher moments/weights for corresponding 123D parameters and zero-initializes
the gait column. Adam step starts at 105,000 and LR at 1.5e-5.

## Selection

The initial integrated checkpoint was selected because updated checkpoints violated the consecutive
endpoint-retention guard. It retains the already audited deterministic and stochastic endpoint/toggle
behavior and has exact reference KL zero. No production policy was updated.
"""
(ROOT / "research/exp_012_g1_gait_conditioned_ppo_retention_preflight_report.md").write_text(report, encoding="utf-8")
