"""Finalize the fail-closed A7-R1 result after the replay identity gate."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
BASE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
OUT = BASE / "phase_w2_p1_a7_r1_rear_yaw_start_teacher_masked_ppo"
REPORT = REPO / "research/exp_013_g1_phase_w2_p1_a7_r1_rear_yaw_start_teacher_masked_ppo_report.md"
CLASSIFICATION = "EXP013_W2_P1_A7_R1_MASK_CONTRACT_IDENTITY_FAIL"
START = "7308c30e5f7a92dc74aba28f25f7991b68f5e2ec"
POOL_SHA = "1397a99c6fb8975c43b6f951ee82432a1d543e13ea94a7991bd7373bf8544853"
MASK_SHA = "0e32a2b41eae4996c1ec6acf7ef929c473af76e9685e14c2f12f738e1b9e6fb6"


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


identity = json.loads((OUT / "a7_full_batch_replay_identity.json").read_text(encoding="utf-8"))
identity.update({
    "fresh_launch_attempts": 3,
    "mismatch_set_repeatable": True,
    "identity_gate": "FAIL",
    "persistent_PPO_updates": 0,
    "policy_parameter_updates": 0,
    "failure_scope": "batch 0 formal-stop replay before actor switch",
})
dump("a7_full_batch_replay_identity.json", identity)

mask_audit = json.loads((OUT / "a7_mask_contract_identity_audit.json").read_text(encoding="utf-8"))
mask_audit.update({
    "m0_static_artifact_contract": "PASS",
    "r1_live_full_batch_identity": "FAIL",
    "r1_live_actual_accepted": identity["actual_accepted"],
    "r1_live_expected_selected": identity["expected_accepted"],
    "r1_live_mismatch_count": identity["mismatch_count"],
    "status": "FAIL",
})
dump("a7_mask_contract_identity_audit.json", mask_audit)

not_executed = {
    "status": "NOT_EXECUTED",
    "reason": CLASSIFICATION,
    "identity_gate_stage": "before PPO collection and optimizer update",
    "optimizer_updates": 0,
}
dump("a7_rollin_ppo_separation_audit.json", {
    **not_executed,
    "teacher_rollin_samples_in_PPO": 0,
    "validation_samples_in_PPO": 0,
    "heldout_samples_in_PPO": 0,
    "rejected_or_unselected_samples_in_PPO": 0,
})
dump("a7_simulator_step_accounting.json", {
    "identity_diagnostic_fresh_launches": 3,
    "identity_diagnostic_teacher_rollin_env_steps_per_launch": 153600,
    "persistent_run_teacher_rollin_steps": 0,
    "policy_prefix_warmup_steps": 0,
    "PPO_valid_train_samples": 0,
    "masked_invalid_post_switch_steps": 0,
    "housekeeping_post_switch_env_steps": 0,
    "persistent_training_total_simulator_steps": 0,
    "optimizer_updates": 0,
})

write_csv("a7_parent_start_baseline.csv", [{"status": "NOT_EXECUTED", "reason": CLASSIFICATION, "episodes": 0}])
dump("a7_parent_start_baseline.json", {**not_executed, "conditions": 0, "A6_live_comparison": "NOT_EXECUTED"})
dump("first_update_stability.json", {**not_executed, "effective_valid_samples": 0, "exact_KL": None, "clip_fraction": None})
dump("early_guard.json", {**not_executed, "updates_evaluated": 0})
write_csv("training_curves.csv", [{"update": 0, "phase": "INITIAL", "optimizer_update": 0, "status": "BLOCKED_BY_IDENTITY_GATE"}])
dump("checkpoint_manifest.json", {**not_executed, "checkpoints": [], "new_policy_checkpoints": 0})
write_csv("a7_capability_timeline.csv", [{"update": 0, "status": "NOT_EVALUATED", "reason": CLASSIFICATION}])
dump("a7_capability_timeline.json", {**not_executed, "timeline": []})
dump("selected_checkpoint.json", {**not_executed, "selected_checkpoint": None, "fallback": False})
dump("selected_checkpoint_process_parity.json", {**not_executed, "selected_checkpoint": None})
write_csv("formal_start_matrix.csv", [{"status": "NOT_EXECUTED", "conditions": 0, "episodes": 0}])
dump("formal_start_matrix.json", {**not_executed, "conditions": 0})
write_csv("formal_pure_yaw_start.csv", [{"status": "NOT_EXECUTED", "conditions": 0, "episodes": 0}])
dump("formal_pure_yaw_start.json", {**not_executed, "conditions": 0})
write_csv("formal_rear_speed_boundary.csv", [{"status": "NOT_EXECUTED", "conditions": 0, "episodes": 0}])
dump("formal_rear_speed_boundary.json", {**not_executed, "conditions": 0})
dump("safety_summary.json", {**not_executed, "fall": None, "slip": None, "impact": None, "saturation": None})
dump("rear_start_symmetry.json", {**not_executed, "endpoint_difference_pp": None, "acquisition_difference_pp": None})
dump("single_teacher_audit.json", {
    "status": "PASS_NO_TEACHER_CREATED",
    "unique_new_checkpoint": 0,
    "unique_new_actor": 0,
    "runtime_teacher": 0,
    "runtime_expert": 0,
    "router": 0,
    "checkpoint_switch": 0,
    "action_blending": 0,
})

protected = [
    BASE / "phase_w2_p1_a7_s0_formal_stop_state_pool/a7_stop_initialization_authorization.json",
    BASE / "phase_w2_p1_a7_s0_formal_stop_state_pool/formal_stop_replay_recipe_manifest.json",
    BASE / "phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight/a7_masked_ppo_training_authorization.json",
    BASE / "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt",
]
dump("protected_hashes.json", {
    "files": [{"path": str(path.relative_to(REPO)).replace("\\", "/"), "sha256": sha(path)} for path in protected],
    "existing_datasets_changed": 0,
    "existing_labels_changed": 0,
    "existing_splits_changed": 0,
    "existing_manifests_changed": 0,
    "existing_overlays_changed": 0,
    "existing_checkpoints_changed": 0,
    "existing_optimizers_changed": 0,
    "reward_changed": 0,
    "physics_changed": 0,
    "new_policy_checkpoint": 0,
    "canonical_runtime_promotion": 0,
    "remote_push": False,
})
dump("stage_classification.json", {"classification": CLASSIFICATION, "primary": True})
dump("recommended_next_action.json", {
    "action": "repair and independently reauthorize the full-batch replay identity contract before any A7-R1 PPO run",
    "rerun_A7_R1_authorized": False,
})
dump("gate.json", {
    "classification": CLASSIFICATION,
    "identity_gate": "FAIL",
    "persistent_PPO_started": False,
    "optimizer_updates": 0,
    "formal_evaluation": 0,
    "teacher_artifact_created": False,
    "canonical_promotion": 0,
    "remote_push": False,
})
(OUT / "reproduction_commands.ps1").write_text(
    '& "$env:USERPROFILE\\workspace\\IsaacLab\\isaaclab.bat" -p '
    'experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/'
    'train_w2_p1_a7_r1_masked.py --updates 1 --headless --device cuda:0\n',
    encoding="utf-8",
)

REPORT.write_text(f"""# Exp 013 Phase W2-P1-A7-R1 rear-yaw start teacher masked-PPO run

Classification: `{CLASSIFICATION}`.

The mandatory identity gate stopped the run before PPO collection or any optimizer update. Three fresh launches reproduced the same batch-0 mismatch: the live replay selected {identity['actual_accepted']} formal-stop environments while the authorized M0 mask selected {identity['expected_accepted']}, with {identity['mismatch_count']} environment identities disagreeing. The differences are physical, not threshold rounding. For example, environment 207 changed from M0 speed/yaw 0.00947/0.00234 and PASS to R1 speed/yaw 0.37568/0.71188 with fall and slip.

Because the authorized contract requires per-state semantic identity and exact accepted IDs before the persistent run, no policy update, training checkpoint, checkpoint selection, held-out evaluation, or rear-yaw teacher artifact was produced. The parent, S0 replay recipe, M0 masks, datasets, labels, overlays, reward, and physics remain unchanged. Snapshot restore and unmasked PPO were not used.
""", encoding="utf-8")

dump("stage_reference.json", {
    "stage": "W2-P1-A7-R1",
    "starting_head": START,
    "actual_starting_head": START,
    "pre_commit_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
})
print(json.dumps({"classification": CLASSIFICATION, "optimizer_updates": 0, "mismatch_count": identity["mismatch_count"]}, indent=2))
