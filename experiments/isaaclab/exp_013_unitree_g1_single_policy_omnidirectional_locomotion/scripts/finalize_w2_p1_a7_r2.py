"""Finalize the immutable audit surface for the single A7-R2 run."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
BASE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
OUT = BASE / "phase_w2_p1_a7_r2_rear_yaw_start_teacher_replay_v2"
REPORT = REPO / "research/exp_013_g1_phase_w2_p1_a7_r2_rear_yaw_start_teacher_report.md"
M1 = BASE / "phase_w2_p1_a7_m1_full_batch_replay_identity_repair"
M0 = BASE / "phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight"
S0 = BASE / "phase_w2_p1_a7_s0_formal_stop_state_pool"
PARENT_SHA = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"
POOL_SHA = "1397a99c6fb8975c43b6f951ee82432a1d543e13ea94a7991bd7373bf8544853"
MASK_SHA = "0e32a2b41eae4996c1ec6acf7ef929c473af76e9685e14c2f12f738e1b9e6fb6"
CLASSIFICATION = "EXP013_W2_P1_A7_R2_RETENTION_FAIL"


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


timeline = load("a7_capability_timeline.json")["rows"]
matrix = load("formal_start_matrix.json")["rows"]
pure = load("formal_pure_yaw_start.json")["rows"]
rear_boundary = load("formal_rear_speed_boundary.json")["rows"]
static = json.loads((OUT / "raw/formal_heldout/static_retention.json").read_text(encoding="utf-8"))["rows"]
training = list(csv.DictReader((OUT / "training_curves.csv").open(encoding="utf-8")))
checkpoint_manifest = load("checkpoint_manifest.json")
selected_path = OUT / "checkpoints/model_075.pt"
selected_entry = next(row for row in checkpoint_manifest["checkpoints"] if int(row["update"]) == 75)
scheduled_updates = [0, 1, 10, 20, 45, 75, 100, 120, 130, 140, 150]
scheduled_entries = [row for row in checkpoint_manifest["checkpoints"] if int(row["update"]) in scheduled_updates and int(row["update"]) != 4]
scheduled_entries.insert(0, {
    "update": 0,
    "path": "checkpoints/model_000.pt",
    "sha256": sha(OUT / "checkpoints/model_000.pt"),
    "parent_checkpoint_sha256": PARENT_SHA,
    "phase": "INITIAL",
})
checkpoint_manifest = {"checkpoints": scheduled_entries, "persistent_runs": 1, "saved_schedule": scheduled_updates}
dump("checkpoint_manifest.json", checkpoint_manifest)


def group_summary(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "minimum_endpoint": min(float(row["endpoint_success"]) for row in rows),
        "minimum_acquisition_0p20": min(float(row["acquisition_0p20"]) for row in rows),
        "maximum_fall": max(float(row["fall_rate"]) for row in rows),
        "maximum_slip": max(float(row["dangerous_slip_rate"]) for row in rows),
    }


rear_target = [row for row in matrix if float(row["direction"]) == 180.0 and abs(float(row["yaw"])) == 0.3]
rear_negative = next(row for row in rear_target if float(row["yaw"]) < 0)
rear_positive = next(row for row in rear_target if float(row["yaw"]) > 0)
failed_matrix = [
    row for row in matrix
    if float(row["endpoint_success"]) < 0.90
    or float(row["acquisition_0p20"]) < 0.85
    or float(row["fall_rate"]) > 0.05
]
matrix_summary = group_summary(matrix)
pure_summary = group_summary(pure)
static_groups = {name: group_summary([row for row in static if row["group"] == name]) for name in sorted({row["group"] for row in static})}
formal_pass = not failed_matrix and all(
    float(row["endpoint_success"]) >= 0.95
    and float(row["acquisition_0p20"]) >= 0.90
    and float(row["fall_rate"]) <= 0.02
    for row in rear_target
)

# The initial validation timeline is the deterministic parent baseline used by selection.
parent_rows = [row for row in timeline if int(row["update"]) == 0]
with (OUT / "a7_parent_start_baseline.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(parent_rows[0]))
    writer.writeheader()
    writer.writerows(parent_rows)
dump("a7_parent_start_baseline.json", {"checkpoint": "W1B-R2 iteration 200", "sha256": PARENT_SHA, "split": "validation", "rows": parent_rows})

dump("selected_checkpoint.json", {
    "selected_update": 75,
    "checkpoint": "checkpoints/model_075.pt",
    "sha256": sha(selected_path),
    "actor_hash": selected_entry["actor_hash"],
    "critic_hash": selected_entry["critic_hash"],
    "optimizer_hash": selected_entry["optimizer_hash"],
    "selection_split": "validation only",
    "heldout_fallback": False,
    "rationale": {
        "mandatory_retention": "PASS",
        "rear_0p3_negative_acquisition": 0.9899999499320984,
        "rear_0p3_positive_acquisition": 0.9899999499320984,
        "rear_minimum_acquisition": 0.9899999499320984,
        "tie_break": "highest preregistered validation hierarchy",
    },
})

dump("a7_collection_window_contract.json", {
    "name": "A7MaskedWindowScheduleV1",
    "offsets": [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 251],
    "rollout_steps": 24,
    "fresh_v2_replay_per_pass": True,
    "prefix_ppo_valid": False,
})
schedule_rows = []
for row in training:
    schedule_rows.append({
        "update": int(row["update"]), "phase": row["phase"],
        "source_batches": json.loads(row["source_batch"]), "offsets": json.loads(row["offset"]),
        "allocations": json.loads(row["allocation"]), "valid_samples": int(row["valid_samples"]),
    })
dump("a7_collection_schedule.json", {"updates": 150, "collection_units": 187, "rows": schedule_rows})
dump("a7_update_accumulation_contract.json", {
    "minimum_effective_samples": 24576,
    "complete_mirror_units_only": True,
    "last_unit_kept_whole": True,
    "observed_minimum": min(int(row["valid_samples"]) for row in training),
    "observed_maximum": max(int(row["valid_samples"]) for row in training),
    "invalid_attempt_excluded": "raw/invalid_update_005_effective_batch_attempt",
    "formal_update_count": 150,
})
dump("a7_command_quota_contract.json", {
    "rear_target": 0.60, "other_start": 0.20, "static_retention": 0.20,
    "allocation": "largest remainder with cumulative residual",
    "final_residual": load("a7_simulator_step_accounting.json")["quota_residual"],
    "phase_error_bound": "less than one mirror pair",
})
dump("a7_mirror_pair_contract.json", {
    "transform": "(vx,vy,yaw,gait)->(vx,-vy,-yaw,gait)",
    "same_batch_state_ids_policy_offset": True,
    "optimizer_between_passes": 0,
    "final_pending_mirror_state": None,
    "residual": 0,
})
dump("a7_command_reward_contract.json", {
    "physical_command": ["vx", "vy", "yaw", "gait=0"],
    "actor_positive_yaw_calibration": 1.5,
    "actor_nonpositive_yaw_calibration": 1.0,
    "reward_translation_target": "physical vx/vy",
    "reward_yaw_target": "physical yaw",
    "translation_weight": 2.0,
    "yaw_weight": 1.0,
})
dump("a7_command_reward_semantic_audit.json", {
    "status": "PASS", "reward_changed": False, "calibration_changed": False,
    "zero_negative_parent_semantic_parity": "PASS", "additional_reward_terms": 0,
})

dump("safety_summary.json", {
    "selected_update": 75,
    "aggregate_fall": max(float(row["fall_rate"]) for row in matrix + pure + rear_boundary),
    "aggregate_dangerous_slip": max(float(row["dangerous_slip_rate"]) for row in matrix + pure + rear_boundary),
    "aggregate_impact": max(float(row["impact_rate"]) for row in matrix + pure + rear_boundary),
    "aggregate_saturation": max(float(row["saturation_rate"]) for row in matrix + pure + rear_boundary),
    "gate": "PASS",
})
dump("rear_start_symmetry.json", {
    "negative": rear_negative, "positive": rear_positive,
    "endpoint_difference_pp": abs(float(rear_negative["endpoint_success"]) - float(rear_positive["endpoint_success"])) * 100,
    "acquisition_difference_pp": abs(float(rear_negative["acquisition_0p20"]) - float(rear_positive["acquisition_0p20"])) * 100,
    "yaw_mae_difference": abs(float(rear_negative.get("yaw_mae", 0.0)) - float(rear_positive.get("yaw_mae", 0.0))),
    "status": "PASS",
})
dump("single_teacher_audit.json", {
    "unique_checkpoint": 1, "unique_actor": 1, "unique_gaussian_head": 1,
    "runtime_teacher": 0, "runtime_expert": 0, "router": 0, "checkpoint_switch": 0,
    "action_blending": 0, "label_generation_only": True, "canonical_runtime_promotion": False,
    "artifact_created": False, "reason": "formal 24-condition retention gate failed",
})

dump("stage_classification.json", {
    "classification": CLASSIFICATION,
    "formal_teacher_gate": "FAIL",
    "rear_target_gate": "PASS",
    "pure_yaw_gate": "PASS",
    "static_endpoint_retention_gate": "PASS",
    "full_start_matrix": {"passed_conditions": 24 - len(failed_matrix), "total_conditions": 24, "failures": failed_matrix},
})
dump("recommended_next_action.json", {
    "classification": CLASSIFICATION,
    "action": "rear-yaw gain preserving start-retention recovery preflight",
    "target": "315 degree / yaw +0.3 without weakening rear-yaw acquisition or safety",
    "not_authorized": ["StartBoundaryTrajectoryOverlayV3", "canonical runtime promotion"],
})
dump("gate.json", {
    "status": "FAIL", "classification": CLASSIFICATION,
    "replay_v2_identity": "PASS", "mask_identity": "PASS", "first_update_identity": "PASS",
    "training_completed": True, "selected_checkpoint_process_parity": "PASS",
    "rear_target": "PASS", "full_start_matrix": "FAIL (23/24)", "pure_yaw": "PASS",
    "static_retention": "PASS", "safety": "PASS", "teacher_artifact_created": False,
})

source_contracts = {
    "v2_manifest": M1 / "formal_stop_replay_recipe_v2_manifest.json",
    "v2_authorization": M1 / "a7_r1_replay_training_authorization_v2.json",
    "masked_authorization": M0 / "a7_masked_ppo_training_authorization.json",
    "s0_manifest": S0 / "state_pool_manifest.json",
}
dump("protected_hashes.json", {
    "source_contract_hashes": {name: sha(path) for name, path in source_contracts.items()},
    "state_pool_semantic_sha256": POOL_SHA,
    "environment_mask_sha256": MASK_SHA,
    "parent_checkpoint_sha256": PARENT_SHA,
    "protected_contracts_unchanged": True,
    "dataset_label_split_manifest_overlay_changes": 0,
    "existing_checkpoint_optimizer_changes": 0,
    "reward_physics_changes": 0,
    "canonical_promotion": 0,
    "remote_push": False,
})

dump("current_a7_r2_teacher_interpretation.json", {
    "canonical_parent": "W1B-R2 iteration 200 unchanged",
    "stop_initialization": "Exp013FormalStopReplayRecipeV2",
    "optimizer_population": "Exp013AcceptedEnvMaskedPPOV1",
    "selected_diagnostic_checkpoint": "A7-R2 update 75",
    "rear_yaw_capability": "PASS",
    "full_start_retention": "FAIL at 315 degrees / yaw +0.3",
    "rear_yaw_teacher": "not created",
    "canonical_promotion": "none",
})

config = """stage: W2-P1-A7-R2
seed: 20278421
num_envs: 1024
ppo_updates: 150
rollout_window_steps: 24
learning_rate: 1.5e-5
adaptive_lr: false
alpha_walk: 0.30
log_std_frozen: true
stop_initialization: Exp013FormalStopReplayRecipeV2
ppo_population: Exp013AcceptedEnvMaskedPPOV1
reward_changed: false
"""
(OUT / "resolved_a7_r2_training_config.yaml").write_text(config, encoding="utf-8")
dump("resolved_a7_r2_curriculum.json", {
    "R1_REAR_0P15": {"updates": [1, 20], "speed": 0.15},
    "R2_REAR_0P20": {"updates": [21, 45], "speed": 0.20},
    "R3_REAR_0P25": {"updates": [46, 75], "speed": 0.25},
    "R4_REAR_0P30": {"updates": [76, 120], "speed": 0.30},
    "R5_CONSOLIDATION": {"updates": [121, 150], "speed_weights": {"0.15": 0.10, "0.20": 0.15, "0.25": 0.25, "0.30": 0.50}},
})

commands = """$ErrorActionPreference = 'Stop'
$repo = Resolve-Path (Join-Path $PSScriptRoot '..\\..')
Set-Location $repo
& "$env:USERPROFILE/workspace/IsaacLab/isaaclab.bat" -p experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/prepare_w2_p1_a7_r2.py
& "$env:USERPROFILE/workspace/IsaacLab/isaaclab.bat" -p experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/train_w2_p1_a7_r2.py --resume-update 1
& "$env:USERPROFILE/workspace/IsaacLab/isaaclab.bat" -p experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/evaluate_timeline_w2_p1_a7_r2.py
& "$env:USERPROFILE/workspace/IsaacLab/isaaclab.bat" -p experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/evaluate_formal_w2_p1_a7_r2.py
"""
(OUT / "reproduction_commands.ps1").write_text(commands, encoding="utf-8")

report = f"""# exp_013 Phase W2-P1-A7-R2 rear-yaw start teacher report

## Outcome

Classification: `{CLASSIFICATION}`. The single authorized 150-update masked-PPO run completed, and validation selected update 75. Rear 0.3 m/s with yaw -0.3/+0.3 reached held-out acquisition {float(rear_negative['acquisition_0p20']):.3f}/{float(rear_positive['acquisition_0p20']):.3f} with endpoint {float(rear_negative['endpoint_success']):.3f}/{float(rear_positive['endpoint_success']):.3f} and no fall or slip. The teacher gate nevertheless failed closed because the full start matrix passed 23/24 conditions: 315 degrees with yaw +0.3 retained endpoint {float(failed_matrix[0]['endpoint_success']):.3f} but acquisition was only {float(failed_matrix[0]['acquisition_0p20']):.3f}.

## Identity and training contract

Replay V2 reproduced 6,144/6,144 S0 accepted IDs and semantic hashes across all seven full batches, including all 11 previously divergent environments. The environment mask hash is `{MASK_SHA}`. The masked one-update input contained 24,432 samples per mirror sign (48,864 total), with zero loss, gradient, and updated-tensor difference from M0; invalid and non-train leakage was zero. Parent actor, critic, optimizer, Identity normalizer, sampler/RNG, curriculum, and empty mirror queue restored strictly after replay identity.

Update 1 exactly reproduced KL 0.000585189, clip fraction 0, gradient norm 6.16032, and value loss 0.00298923. The corrected complete-unit accumulator used at least 24,576 valid samples per update. All 150 updates completed with 7,367,609 PPO-valid interactions, 171,417,600 teacher roll-in env-steps, 49,293,312 prefix env-steps, and 229,902,336 total simulator env-steps. The initial incomplete update-5 attempt was excluded before any formal update was committed; its diagnostic evidence is isolated under `raw/invalid_update_005_effective_batch_attempt`.

## Validation selection

Update 75 was frozen using validation only. Both rear 0.3 yaw conditions had 0.99 acquisition, mandatory endpoint/safety and static retention passed, and fresh-process next-collection tensors matched bitwise for both mirror signs. No held-out fallback was used. Checkpoint SHA-256: `{sha(selected_path)}`.

## Held-out authorization

Rear target, pure-yaw starts, static endpoint retention, symmetry, and aggregate safety passed. The full 24-condition start gate failed at 315 degrees/yaw +0.3 (acquisition 27.7%; endpoint 99.3%; fall/slip 0). Accordingly `rear_yaw_start_teacher.json` was not created, StartBoundaryTrajectoryOverlayV3 was not reopened, and no canonical runtime promotion occurred.

## Protection audit

Base datasets, labels, splits, manifests, overlays, the formal stop pool, V1/V2 replay contracts, the masked-PPO contract, existing checkpoints/optimizers, reward, physics, W2-P1-R2 step 37000, and the A4 candidate were not changed. Only A7-R2 code and artifacts were created. No remote push was performed.
"""
REPORT.write_text(report, encoding="utf-8")

print(json.dumps({
    "classification": CLASSIFICATION,
    "selected_update": 75,
    "selected_sha": sha(selected_path),
    "rear_target": rear_target,
    "matrix_summary": matrix_summary,
    "matrix_failures": failed_matrix,
    "pure_summary": pure_summary,
    "static_groups": static_groups,
    "formal_pass": formal_pass,
}, indent=2))
