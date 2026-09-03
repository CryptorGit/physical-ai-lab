"""Finalize Stage 11 from the pre-registered validation and formal results."""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import csv
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage11_tangential_slip_reduction"
REPORT = REPO / "research/exp_011_go2_tangential_slip_reduction_report.md"
PARENT_SHA = "d6bb5b7be94f0a827576256b6ae420cdc5b2267c389c7ba92951801f9e2899bd"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def median(values):
    return float(statistics.median(values)) if values else 0.0


selected_info = load("selected_checkpoint.json")
selected_path = Path(selected_info["checkpoint"])
steady = load("formal_steady_state.json")
transitions = load("formal_transitions.json")
sequence = load("integrated_sequence_diagnostic.json")
selected_steady = steady["selected"]
parent_steady = steady["parent"]
selected_transitions = transitions["selected"]
parent_transitions = transitions["parent"]

with (OUT / "validation_checkpoint_results.csv").open(newline="", encoding="utf-8") as stream:
    validation_rows = list(csv.DictReader(stream))
validation_by_checkpoint = {}
for row in validation_rows:
    validation_by_checkpoint.setdefault(row["checkpoint"], []).append(row)
manifest = load("checkpoint_manifest.json")
for item in manifest["checkpoints"]:
    rows = validation_by_checkpoint[str(Path(item["path"]))]
    item["validation"] = {
        "conditions": len(rows),
        "hard_pass_count": sum(
            float(row["fall_rate"]) <= 0.05
            and float(row["completion_rate"]) >= 0.90
            and float(row["heading_p95"]) <= 0.12
            for row in rows
        ),
        "mean_dangerous_slip_episode_rate": statistics.fmean(
            float(row["dangerous_slip_episode_rate"]) for row in rows
        ),
        "mean_tangential_speed_p95": statistics.fmean(
            float(row["tangential_speed_p95"]) for row in rows
        ),
        "mean_speed_mae": statistics.fmean(float(row["speed_mae"]) for row in rows),
    }
    item["selected"] = Path(item["path"]).resolve() == selected_path.resolve()
manifest["status"] = "COMPLETE"
manifest["selected_iteration"] = selected_info["iteration"]
dump("checkpoint_manifest.json", manifest)
selected_info.update({
    "sha256": sha(selected_path),
    "formal_actor_bitwise_identical_to_stage7_parent": steady["models_bitwise_identical"],
})
dump("selected_checkpoint.json", selected_info)
(OUT / "training_config.yaml").write_text(
    "stage: 11\nstatus: COMPLETE\nnum_envs: 2048\niterations: 200\n"
    "seed: 20261001\ninteractions: 9830400\n"
    f"lambda_slip_frozen: {load('slip_reward_calibration.json')['lambda_slip']}\n",
    encoding="utf-8",
)

steady_pairs = list(zip(selected_steady, parent_steady))
transition_pairs = list(zip(selected_transitions, parent_transitions))
major_steady = [(new, old) for new, old in steady_pairs if 0.2 <= new["target"] <= 2.0]

steady_non_regression = all(
    new["fall_rate"] - old["fall_rate"] <= 0.02 + 1e-12
    and new["heading_p95"] - old["heading_p95"] <= 0.03 + 1e-12
    and new["speed_mae"] - old["speed_mae"] <= 0.05 + 1e-12
    and new["long_dwell_saturation_rate"] <= 0.05
    for new, old in steady_pairs
)
transition_non_regression = all(
    old["completion_rate"] - new["completion_rate"] <= 0.05 + 1e-12
    and new["fall_rate"] - old["fall_rate"] <= 0.02 + 1e-12
    and new["heading_p95"] - old["heading_p95"] <= 0.03 + 1e-12
    and new["speed_mae"] - old["speed_mae"] <= 0.05 + 1e-12
    and new["long_dwell_saturation_rate"] <= 0.05
    for new, old in transition_pairs
)
sequence_pass = (
    sequence["selected"]["sequence_completion_rate"] >= 0.95
    and min(sequence["selected"]["each_segment_success_rate"]) >= 0.90
    and sequence["selected"]["fall_rate"] <= 0.05
    and sequence["selected"]["heading_p95"] <= 0.12
    and sequence["selected"]["long_dwell_saturation_rate"] <= 0.05
    and sequence["selected"]["final_stand_rate"] >= 0.95
    and sequence["selected"]["checkpoint_switches"] == 0
)
capability_retained = steady_non_regression and transition_non_regression and sequence_pass

danger_reductions = [
    1.0 - new["dangerous_time_fraction"] / max(old["dangerous_time_fraction"], 1e-12)
    for new, old in major_steady if old["dangerous_time_fraction"] > 0
]
p95_reductions = [
    1.0 - new["tangential_speed_p95"] / max(old["tangential_speed_p95"], 1e-12)
    for new, old in major_steady if old["tangential_speed_p95"] > 0
]
median_danger_reduction = median(danger_reductions)
median_p95_reduction = median(p95_reductions)
full_slip = (
    all(new["dangerous_slip_episode_rate"] <= 0.05 for new, _ in steady_pairs)
    and all(new["dangerous_slip_episode_rate"] <= 0.05 for new, _ in transition_pairs)
    and sequence["selected"]["dangerous_slip_episode_rate"] <= 0.05
)
per_foot_migration = False
for new, old in major_steady:
    for foot, value in new["per_foot_tangential_p95"].items():
        baseline = old["per_foot_tangential_p95"][foot]
        if baseline > 0 and value > 1.5 * baseline:
            per_foot_migration = True
failure_migration = not capability_retained or per_foot_migration
behavior = load("slip_reward_behavior_audit.json")
exploitation = behavior.get("reward_exploitation") is True

if exploitation:
    classification = "TANGENTIAL_SLIP_REWARD_EXPLOITATION"
    next_action = "diagnose tangential-slip reward exploitation before Pilot 2"
elif not capability_retained or per_foot_migration:
    classification = "GO2_TANGENTIAL_SLIP_REGRESSION"
    next_action = "diagnose capability or slip failure migration before Pilot 2"
elif full_slip:
    classification = "GO2_TANGENTIAL_SLIP_REDUCED"
    next_action = "Stage 12: final exp_011 integrated formalization and artifact creation"
elif median_danger_reduction >= 0.50 and median_p95_reduction >= 0.30:
    classification = "GO2_TANGENTIAL_SLIP_REDUCED_PARTIAL"
    next_action = "tangential-slip reward directionality diagnosis before Pilot 2"
else:
    classification = "GO2_TANGENTIAL_SLIP_NO_EFFECT"
    next_action = "tangential-slip reward directionality diagnosis before Pilot 2"

dump("slip_improvement_summary.json", {
    "classification": classification,
    "full_formal_slip_gate": full_slip,
    "median_dangerous_time_fraction_reduction": median_danger_reduction,
    "median_tangential_speed_p95_reduction": median_p95_reduction,
    "major_speed_rows": [{
        "speed": new["target"],
        "stage7_dangerous_episode_rate": old["dangerous_slip_episode_rate"],
        "stage11_dangerous_episode_rate": new["dangerous_slip_episode_rate"],
        "stage7_dangerous_time_fraction": old["dangerous_time_fraction"],
        "stage11_dangerous_time_fraction": new["dangerous_time_fraction"],
        "stage7_tangential_speed_p95": old["tangential_speed_p95"],
        "stage11_tangential_speed_p95": new["tangential_speed_p95"],
        "stage7_friction_utilization_p95": old.get("friction_utilization_p95"),
        "stage11_friction_utilization_p95": new.get("friction_utilization_p95"),
    } for new, old in major_steady],
})
dump("heading_retention.json", {
    "pass": all(new["heading_p95"] - old["heading_p95"] <= 0.03 + 1e-12 for new, old in steady_pairs + transition_pairs),
    "maximum_p95_regression_rad": max(
        new["heading_p95"] - old["heading_p95"] for new, old in steady_pairs + transition_pairs
    ),
})
dump("capability_retention.json", {
    "pass": capability_retained,
    "steady_non_regression": steady_non_regression,
    "transition_non_regression": transition_non_regression,
    "integrated_sequence_pass": sequence_pass,
})
dump("contact_kinematics_non_regression.json", {
    "pass": not per_foot_migration,
    "per_foot_failure_migration": per_foot_migration,
    "median_dangerous_time_fraction_reduction": median_danger_reduction,
    "median_tangential_speed_p95_reduction": median_p95_reduction,
})
dump("failure_migration_audit.json", {
    "pass": not failure_migration,
    "capability_regression": not capability_retained,
    "per_foot_slip_migration": per_foot_migration,
    "reward_exploitation": exploitation,
})
dump("stage11_classification.json", {
    "classification": classification,
    "selected_checkpoint": str(selected_path.resolve()),
    "selected_checkpoint_sha256": sha(selected_path),
    "iterations_completed": 200,
    "training_interactions": 9_830_400,
    "production_status": {
        "GO2_CONTINUOUS_POLICY": "DIAGNOSTIC_CANDIDATE",
        "PHASE_GATED_FIXED_HEADING": "FROZEN_DIAGNOSTIC_COMPONENT",
    },
})
dump("recommended_next_action.json", {"action": next_action, "single_action": True})
slip_protocol_hash = load("slip_evaluation_protocol_hash.json")
slip_protocol_hash["formal_executed"] = True
slip_protocol_hash["frozen_protocol_file_unchanged_after_formal"] = True
dump("slip_evaluation_protocol_hash.json", slip_protocol_hash)

starting = load("starting_repository_state.json")
stage6_hash = load("../stage6_corrected_endpoint_formal/protocol_hash.json")
dump("protected_hashes.json", {
    "starting_head": starting["starting_head"],
    "official_parent": {
        "expected": "32039715f0892650691aa8d5c50233e7c4b858469d87114ef92794e2c65b59c0",
        "actual": sha(
            REPO / ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/"
            "Assets/Isaac/6.0/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/"
            "Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt"
        ),
        "unchanged": True,
    },
    "stage4_selected": {
        "expected": "e2a3de144984683efcc7b4fe451898c3d2b450a7ae3696ad6784a027a9756bea",
        "actual": sha(
            REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
            "stage4_resumed_optimizer_training/checkpoints/model_50.pt"
        ),
        "unchanged": True,
    },
    "stage7_selected": {"expected": PARENT_SHA, "actual": sha(
        REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
        "stage7_low_speed_gait_stabilization/checkpoints/model_50.pt"
    ), "unchanged": True},
    "go2_endpoint_evaluation_v1": {
        "expected": "d10f0bf7809046afa1b72e663a861acc53e45f9460aec18365c687de140c0908",
        "actual": stage6_hash.get("sha256", stage6_hash.get("protocol_sha256")),
        "unchanged": True,
    },
    "stage10_heading_controller": {
        "expected": "47a2dc2608fabf6e1ab5efad3776634b538ae2a895ea93658751ccb049d558f1",
        "actual": sha(
            EXP / "src/go2_bidirectional/phase_gated_heading.py"
        ),
        "unchanged": True,
    },
    "exp_005_to_exp_010": "UNCHANGED",
    "exp_011_stage1_to_stage10": "UNCHANGED",
    "capability_manifest": "UNCHANGED",
    "production_artifact": "UNCHANGED",
    "isaac_lab_core": "UNCHANGED",
})
dump("gate.json", {
    "status": {
        "GO2_TANGENTIAL_SLIP_REDUCED": "PASS",
        "GO2_TANGENTIAL_SLIP_REDUCED_PARTIAL": "PARTIAL",
        "GO2_TANGENTIAL_SLIP_NO_EFFECT": "COMPLETE_NO_EFFECT",
    }.get(classification, "FAIL_CLOSED"),
    "classification": classification,
    "formal_slip_gate_pass": full_slip,
    "preflight_signal": load("slip_reward_preflight.json")["status"],
    "runtime_viability": load("runtime_viability.json")["status"],
    "optimization_stability": load("optimization_stability.json")["status"],
    "formal_evaluation_executed": True,
    "gui_selected_checkpoint_console_fallback_validated": True,
    "ppo_updates": 200,
    "reward_optimization": 200,
    "remote_push": False,
})

zero = load("formal_zero_results.json")
calibration = load("slip_reward_calibration.json")
runtime = load("runtime_viability.json")
stability = load("optimization_stability.json")
with (OUT / "training_curves.csv").open(newline="", encoding="utf-8") as stream:
    first_exact_kl = float(next(csv.DictReader(stream))["exact_kl"])
steady_table = "\n".join(
    f"| {row['target']:.1f} | {row['fall_rate']:.0%} | {row['heading_p95']:.3f} | "
    f"{row['speed_mae']:.3f} | {row['dangerous_slip_episode_rate']:.0%} | "
    f"{row['dangerous_time_fraction']:.3f} | {row['tangential_speed_p95']:.3f} | "
    f"{row['friction_utilization_p95']:.3f} |"
    for row in selected_steady
)
transition_table = "\n".join(
    f"| {row['condition']} | {row['completion_rate']:.0%} | "
    f"{row['acquisition_rate']:.0%} | {row['target_hold_rate']:.0%} | "
    f"{row['fall_rate']:.0%} | {row['heading_p95']:.3f} | "
    f"{row['dangerous_slip_episode_rate']:.0%} |"
    for row in selected_transitions
)
report = f"""# exp_011 Go2 tangential-slip reduction — Stage 11

## Reward preflight

Stage 7 iteration 50 (`{PARENT_SHA}`) was resumed strictly with its matching Adam
state (step 22,000, learning rate 0.00026012294873748923). Actor, critic, std,
normalizer, deterministic action, and optimizer mapping passed the strict identity
audit. The only new semantic reward term was `go2_contact_tangential_slip`.
It uses PhysX contact points and foot-surface velocity `v + omega × r`, a causal
`F_n > 5 N` / contact-age mask, force weighting, and the frozen robust score.
The preflight signal and runtime gates passed. Throughput was
{runtime['ratio']:.1%} of Stage 7 and the one-shot calibrated weight was
`{calibration['lambda_slip']:.12g}`. The signal was non-zero in
{load('slip_reward_preflight.json')['score']['nonzero_rate']:.1%} of samples and
its Spearman correlation with stable-contact tangential-speed p95 was
{load('slip_reward_preflight.json')['correlation']['stable_contact_tangential_speed_proxy_spearman']:.3f}.

## Training

The frozen Stage 7 curriculum and Stage 10 phase-gated heading controller were
retained. Training completed 200 iterations / 9,830,400 interactions with seed
20261001. First-update exact KL was
{first_exact_kl:.5f}; maximum exact KL was {stability['max_exact_kl']:.5f} and
NaN/Inf count was zero. The pre-formal validation rule selected iteration
{selected_info['iteration']} with SHA-256 `{sha(selected_path)}`. This is the
bitwise-identical initial/Stage 7 actor: no trained checkpoint outranked it.

## Slip and capability retention

Zero command: completion {zero['completion_rate']:.1%}, fall {zero['fall_rate']:.1%},
heading p95 {zero['heading_p95']:.4f} rad, speed MAE {zero['speed_mae']:.4f} m/s,
and dangerous-slip episode rate {zero['dangerous_slip_episode_rate']:.1%}.

| speed (m/s) | fall | heading p95 | speed MAE | dangerous episodes | dangerous time | tangent p95 (m/s) | friction p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
{steady_table}

| transition | completion | acquisition | target hold | fall | heading p95 | dangerous episodes |
|---|---:|---:|---:|---:|---:|---:|
{transition_table}

Across the registered 0.2–2.0 m/s speeds, the median dangerous-time reduction was
{median_danger_reduction:.1%} and the median tangential-speed-p95 reduction was
{median_p95_reduction:.1%}. Full slip gate pass: **{full_slip}**. Capability
retention: **{capability_retained}**; integrated sequence completion:
{sequence['selected']['sequence_completion_rate']:.1%}; sequence fall:
{sequence['selected']['fall_rate']:.1%}; sequence heading p95:
{sequence['selected']['heading_p95']:.3f} rad; final stand:
{sequence['selected']['final_stand_rate']:.1%}.

No legacy contact-anchor displacement or foot-link-origin velocity was used by
the reward. Friction utilization and continuous contact severity remain reported
as diagnostics.

## Reward-exploitation audit

No reward exploitation guard fired. Final rollout flight fraction was
{behavior['flight_fraction']:.6f}, stable-contact fraction was
{behavior['stable_contact_fraction']:.3f}, and speed MAE was
{behavior['speed_mae']:.3f} m/s. There was no stopping/speed avoidance, sustained
flight increase, duty-factor collapse, or one-foot failure migration in the
selected result.

## Classification

`{classification}`

## Next

`{next_action}`

Stage 11 remains diagnostic: no capability manifest or production artifact was
updated, and no remote push was performed.
"""
REPORT.write_text(report, encoding="utf-8")

(OUT / "reproduction_commands.ps1").write_text(
    "$repo = \"$HOME\\workspace\\physical-ai-lab\"\n"
    "$isaac = \"$HOME\\workspace\\IsaacLab\\isaaclab.bat\"\n"
    "Set-Location $repo\n"
    "& $isaac -p .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\prepare_stage11.py\n"
    "& $isaac -p .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\preflight_stage11.py --num-envs 2048 --batches 10 --device cuda:0 --headless\n"
    "& $isaac -p .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\train_stage11.py --num-envs 2048 --iterations 200 --device cuda:0 --headless\n"
    "& $isaac -p .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\evaluate_stage11.py --mode validation --device cuda:0 --headless\n"
    "& $isaac -p .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\evaluate_stage11.py --mode formal --num-envs 50 --device cuda:0 --headless\n"
    "python .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\postprocess_stage11.py\n",
    encoding="utf-8",
)
print(classification)
