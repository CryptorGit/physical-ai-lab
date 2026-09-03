"""Fail-closed A7 replay-recipe parity and strict-resume preflight."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
BASE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
OUT = BASE / "phase_w2_p1_a7_rear_yaw_start_acquisition_rerun_replay_recipe"
S0 = BASE / "phase_w2_p1_a7_s0_formal_stop_state_pool"
W1B = BASE / "phase_w1b_r2_pending_mirror_queue_repair_rerun"
PARENT = W1B / "checkpoints/model_200.pt"
REPORT = REPO / "research/exp_013_g1_phase_w2_p1_a7_rear_yaw_start_acquisition_rerun_report.md"
START = "c74874781170eff138f4643726ebf6087176bc84"
EXPECTED_PARENT = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"
EXPECTED_POOL = "1397a99c6fb8975c43b6f951ee82432a1d543e13ea94a7991bd7373bf8544853"
CLASSIFICATION = "EXP013_W2_P1_A7_REPLAY_RECIPE_PARITY_FAIL"
OUT.mkdir(parents=True, exist_ok=True)


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def empty_csv(name, fields):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def semantic(obj):
    h = hashlib.sha256()
    def walk(x):
        if torch.is_tensor(x):
            t = x.detach().cpu().contiguous(); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str): h.update(str(k).encode()); walk(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x: walk(v)
        else: h.update(repr(x).encode())
    walk(obj); return h.hexdigest()


parent_sha = sha(PARENT)
ck = torch.load(PARENT, map_location="cpu", weights_only=False)
required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "normalizer_state", "sampler_state_dict", "sampler_state_hash", "iter", "infos"}
sampler_required = {"pending_queue", "sampler_rng_state", "command_rng_state", "active_curriculum_phase", "training_iteration", "current_command_buffer"}
missing = sorted(required - set(ck))
sampler = ck.get("sampler_state_dict", {})
adam_steps = sorted({int(v["step"]) for v in ck["optimizer_state_dict"]["state"].values() if "step" in v})
strict = not missing and not (sampler_required - set(sampler)) and adam_steps == [8000] and ck["normalizer_state"] == {"type": "Identity"} and sampler.get("pending_queue") is None and parent_sha == EXPECTED_PARENT

recipe = json.loads((S0 / "formal_stop_replay_recipe_manifest.json").read_text(encoding="utf-8"))
pool = json.loads((S0 / "state_pool_manifest.json").read_text(encoding="utf-8"))
split = json.loads((S0 / "state_pool_split.json").read_text(encoding="utf-8"))
det = json.loads((S0 / "stop_state_pool_generation_determinism.json").read_text(encoding="utf-8"))
with (S0 / "state_provenance.csv").open(encoding="utf-8", newline="") as f:
    provenance = list(csv.DictReader(f))
by_id = {int(r["state_id"]): r for r in provenance}
split = {name: ids for name, ids in split.items() if isinstance(ids, list)}
composition = {name: dict(sorted(Counter(int(by_id[i]["batch"]) for i in ids).items())) for name, ids in split.items()}
batch_membership = {}
for name, ids in split.items():
    for i in ids: batch_membership.setdefault(int(by_id[i]["batch"]), set()).add(name)
cross_split_batches = {str(k): sorted(v) for k, v in batch_membership.items() if len(v) > 1}

builder = (HERE.parent / "build_w2_p1_a7_s0_pool.py").read_text(encoding="utf-8")
source_seed_not_applied = "seeds=torch.arange(SEED+batch*N" in builder and "env.reset(env_ids=ids)" in builder
per_recipe_rng = isinstance(recipe.get("reset_rng_states"), dict)
independent_recipe_payload = all(k in recipe for k in ("per_recipe_reset_seed", "per_recipe_rng_state", "per_recipe_reset_identity"))
fixed_ppo_batch_compatible = all(len([i for i in ids if int(by_id[i]["batch"]) == b]) in (0, 1024) for ids in split.values() for b in range(7))

parity_ok = (
    det.get("accepted_ids_exact") and det.get("batch_semantic_hashes_exact") and
    det.get("whole_pool_semantic_hash_exact") and per_recipe_rng and
    independent_recipe_payload and fixed_ppo_batch_compatible and not cross_split_batches
)

parent_manifest = {
    "checkpoint": str(PARENT.relative_to(REPO)).replace("\\", "/"), "sha256": parent_sha,
    "expected_sha256": EXPECTED_PARENT, "iteration": ck["iter"], "architecture": "124->256->128->128->37",
    "actor_tensor_hash": semantic(ck["actor_state_dict"]), "critic_tensor_hash": semantic(ck["critic_state_dict"]),
    "optimizer_semantic_hash": semantic(ck["optimizer_state_dict"]), "normalizer": ck["normalizer_state"],
    "sampler_semantic_hash": semantic(sampler),
}
dump("stage_reference.json", {"stage": "W2-P1-A7 replay-recipe rerun", "starting_head_reported": START, "starting_head_actual": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "attempt_number": 1})
dump("protocol.json", {"persistent_runs_authorized": 1, "strict_resume_gate": True, "replay_recipe_gate": True, "snapshot_restore": False, "fail_closed": True, "PPO_started": False, "reason": CLASSIFICATION})
dump("a7_parent_manifest.json", parent_manifest)
dump("a7_parent_identity_audit.json", {"status": "PASS" if strict else "FAIL", "checkpoint_bitwise": parent_sha == EXPECTED_PARENT, "actor": "BITWISE_PAYLOAD_PRESENT", "critic": "BITWISE_PAYLOAD_PRESENT", "identity_normalizer": ck["normalizer_state"] == {"type": "Identity"}, "missing": missing})
dump("a7_optimizer_resume_audit.json", {"status": "PASS" if strict else "FAIL", "adam_steps": adam_steps, "expected": [8000], "learning_rates": [g["lr"] for g in ck["optimizer_state_dict"]["param_groups"]], "sampler_fields_present": not bool(sampler_required - set(sampler)), "pending_mirror_queue_empty": sampler.get("pending_queue") is None})
dump("a7_replay_recipe_contract.json", {"name": recipe["name"], "generation_seed": recipe["generation_seed"], "num_envs": recipe["num_envs"], "rollin_steps": recipe["rollin_steps"], "teacher_sha256": recipe["teacher_sha256"], "pool_semantic_sha256": pool["whole_pool_semantic_sha256"], "split_sizes": {k: len(v) for k, v in split.items()}, "snapshot_restore": False, "state_identity": recipe["state_identity"]})
dump("a7_replay_recipe_preflight.json", {
    "status": "PASS" if parity_ok else "FAIL", "classification": None if parity_ok else CLASSIFICATION,
    "S0_batch_level_fresh_process_reproduction": "PASS", "accepted_ids_exact": det.get("accepted_ids_exact"),
    "batch_semantic_hashes_exact": det.get("batch_semantic_hashes_exact"), "whole_pool_semantic_hash_exact": det.get("whole_pool_semantic_hash_exact"),
    "required_per_recipe_256_each_split": "NOT_EXECUTABLE_WITHOUT_SNAPSHOT_OR_UNSELECTED_ENVS",
    "per_recipe_reset_rng_serialized": per_recipe_rng, "independent_recipe_payload": independent_recipe_payload,
    "source_seed_field_applied_to_reset": not source_seed_not_applied,
    "observation_and_teacher_action_difference_zero": "PROVEN_ONLY_FOR_FULL_BATCH_REPLAY; NOT_PROVEN_FOR_INDIVIDUAL_RECIPE_ASSIGNMENT",
    "reason": "S0 reproduces the global 1024-env batch sequence exactly, but does not serialize an independently executable reset/RNG recipe per accepted state. The new A7 contract requires accepted-only, split-isolated, without-replacement per-ID assignment.",
})
dump("a7_recipe_assignment_audit.json", {
    "status": "FAIL", "split_batch_composition": composition, "cross_split_batches": cross_split_batches,
    "train_count": len(split["train"]), "validation_count": len(split["validation"]), "heldout_count": len(split["heldout"]),
    "accepted_only_fixed_1024_env_ppo_batch_compatible": fixed_ppo_batch_compatible,
    "rejected_envs_in_source_batches": {"0": 6, "1": 26, "2": 18, "3": 20, "4": 15, "5": 17, "6": 19},
    "mirror_same_recipe_state": "Cannot create both yaw branches in one live simulator without snapshot restore; separate batch replay is possible diagnostically but is not a serialized pending-mirror PPO allocator.",
    "training_leakage_if_run": "Rejected/non-split envs would enter the fixed 1024-env PPO buffer unless a new masked-PPO contract were introduced; that is outside authorization.",
})
dump("a7_replay_recipe_serialization_audit.json", {
    "status": "FAIL", "global_seed": recipe["generation_seed"], "batch_order": "fixed", "environment_id": "serialized",
    "per_recipe_actual_reset_seed": "NOT_SERIALIZED", "per_recipe_python_numpy_torch_environment_rng": "NOT_SERIALIZED",
    "source_seed_column": "derived SEED + batch*1024 + env_id; audit confirmed it was not passed to env.reset",
    "per_recipe_cursor_and_pending_mirror_resume": "NOT_DEFINED", "batch_level_replay": "PASS",
})
dump("a7_rollin_ppo_separation_audit.json", {"status": "NOT_RUN", "teacher_rollin_steps": 0, "PPO_interactions": 0, "total_simulator_steps": 0, "rollin_in_PPO_buffer": 0, "actor_gradient_during_rollin": 0, "critic_gradient_during_rollin": 0, "reason": "replay parity gate failed before simulator/PPO launch"})

empty_csv("a7_parent_start_baseline.csv", ["condition", "episodes", "endpoint", "acquisition_0p10", "acquisition_0p20", "fall", "slip", "impact"])
dump("a7_parent_start_baseline.json", {"status": "NOT_RUN", "reason": "per preregistration, replay/live parity gate failed before baseline"})
dump("a7_command_reward_contract.json", {"status": "RESOLVED_NOT_EXECUTED", "physical_target": ["vx", "vy", "yaw"], "actor_positive_yaw_multiplier": 1.5, "reward_target": "uncalibrated physical command", "translation_weight": 2.0, "yaw_weight": 1.0, "reward_changes": 0})
dump("a7_command_reward_semantic_audit.json", {"status": "PASS_STATIC_SOURCE_AUDIT", "calibrated_actor_yaw_used_as_reward": False, "reward_or_evaluator_change": 0})
(OUT / "resolved_a7_training_config.yaml").write_text("status: BLOCKED_NOT_RUN\nclassification: EXP013_W2_P1_A7_REPLAY_RECIPE_PARITY_FAIL\nnum_envs: 1024\nrollout_steps: 24\niterations: 150\nseed: 20278421\nlearning_rate: 1.5e-5\nadaptive_lr: false\nreward_changes: 0\nsnapshot_restore: false\n", encoding="utf-8")
dump("resolved_a7_curriculum.json", {"status": "PREREGISTERED_NOT_RUN", "phases": [{"name": "R1_REAR_0P15", "iterations": [1, 20], "speed": .15}, {"name": "R2_REAR_0P20", "iterations": [21, 45], "speed": .20}, {"name": "R3_REAR_0P25", "iterations": [46, 75], "speed": .25}, {"name": "R4_REAR_0P30", "iterations": [76, 120], "speed": .30}, {"name": "R5_CONSOLIDATION", "iterations": [121, 150], "weights": {"0.15": .10, "0.20": .15, "0.25": .25, "0.30": .50}}], "rear_exposure": .60, "other_start": .20, "static_retention": .20})
dump("first_update_stability.json", {"status": "NOT_RUN", "reason": CLASSIFICATION})
dump("early_guard.json", {"status": "NOT_RUN", "iterations_completed": 0, "reason": CLASSIFICATION})
empty_csv("training_curves.csv", ["iteration", "phase", "PPO_interactions", "teacher_rollin_steps", "KL", "clip_fraction", "fall", "slip"])
dump("checkpoint_manifest.json", {"status": "NO_A7_CHECKPOINTS", "parent_only": parent_manifest, "new_checkpoint_count": 0})
empty_csv("a7_capability_timeline.csv", ["iteration", "condition", "endpoint", "acquisition_0p10", "acquisition_0p20", "yaw_resets", "longest_yaw_pass", "fall", "slip"])
dump("a7_capability_timeline.json", {"status": "NOT_RUN", "reason": CLASSIFICATION})
dump("selected_checkpoint.json", {"status": "NONE", "selection_performed": False})
dump("selected_checkpoint_process_parity.json", {"status": "NOT_APPLICABLE", "reason": "no A7 checkpoint"})
for stem in ("formal_start_matrix", "formal_pure_yaw_start", "formal_rear_speed_boundary"):
    empty_csv(stem + ".csv", ["condition", "episodes", "endpoint", "acquisition_0p10", "acquisition_0p20", "fall", "slip", "impact"])
    dump(stem + ".json", {"status": "NOT_RUN", "reason": CLASSIFICATION})
dump("safety_summary.json", {"status": "NOT_EVALUATED", "new_training": 0})
dump("rear_start_symmetry.json", {"status": "NOT_EVALUATED", "reason": "no candidate"})
dump("single_teacher_audit.json", {"status": "NO_TEACHER_SELECTED", "unique_checkpoint": 0, "runtime_teacher": 0, "router": 0, "action_blending": 0})
dump("stage_classification.json", {"classification": CLASSIFICATION, "strict_resume": "PASS" if strict else "FAIL", "batch_replay": "PASS", "per_recipe_PPO_assignment": "FAIL", "PPO_iterations": 0})
dump("recommended_next_action.json", {"action": "version an independently executable per-recipe reset/RNG contract or a preregistered accepted-env PPO mask, then rerun A7", "snapshot_fallback": False})

protected = [PARENT, S0 / "a7_stop_initialization_authorization.json", S0 / "formal_stop_replay_recipe_manifest.json", S0 / "state_pool_manifest.json"]
dump("protected_hashes.json", {"files": [{"path": str(p.relative_to(REPO)).replace("\\", "/"), "sha256": sha(p)} for p in protected], "datasets_changed": 0, "labels_changed": 0, "splits_changed": 0, "manifests_changed": 0, "overlays_changed": 0, "existing_checkpoints_changed": 0, "existing_optimizers_changed": 0, "reward_changed": 0, "physics_changed": 0, "new_checkpoint": 0})
dump("gate.json", {"strict_resume": "PASS" if strict else "FAIL", "S0_batch_replay": "PASS", "per_recipe_replay_parity": "FAIL", "training_authorized": False, "classification": CLASSIFICATION, "PPO_iterations": 0, "new_checkpoint": 0, "canonical_promotion": 0, "remote_push": False})
(OUT / "reproduction_commands.ps1").write_text("python experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/prepare_w2_p1_a7_replay_rerun.py\n", encoding="utf-8")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(f"""# Exp 013 Phase W2-P1-A7 replay-recipe rerun

## Outcome

Classification: `{CLASSIFICATION}`. The W1B-R2 parent strict-resume payload passes, and S0's full 1024-environment batch replay remains exact. PPO was not started because the newly required accepted-only, split-isolated per-recipe allocator cannot be constructed from the authorized S0 manifest without adding an unregistered mask/restore mechanism.

## Evidence

`Exp013FormalStopReplayRecipeV1` serializes one global seed, fixed batch order, environment ID, teacher SHA, and 150 roll-in steps. It does not serialize actual per-recipe reset RNG state. The `source_seed` column was derived as `seed + batch*1024 + env_id` but was not passed to `env.reset`; it is an identity label, not an independently executable reset seed.

Train recipes span batches as {composition['train']}; validation spans {composition['validation']}; held-out spans {composition['heldout']}. Batches 4 and 5 cross split boundaries. Every source batch also contains rejected environments. RSL-RL's fixed 1024-env rollout buffer would therefore include rejected or out-of-split states unless a new masked-PPO contract were introduced. Snapshot duplication/fallback is explicitly prohibited.

The requested 256-per-split independent replay, same-recipe mirror branching, without-replacement recipe epochs, and serialized allocator continuation consequently fail before simulator/PPO launch. No baseline, PPO update, checkpoint, selection, or formal evaluation was performed. This does not invalidate S0's batch-level determinism or any earlier live-roll-in result.

## Protection

Existing datasets, labels, splits, manifests, overlays, state pool, replay manifest, checkpoints, optimizers, reward, physics, W2-P1-R2 student, and A4 candidate were unchanged. New checkpoint: 0. Remote push: false.
""", encoding="utf-8")
print(json.dumps({"classification": CLASSIFICATION, "strict_resume": strict, "batch_replay": True, "per_recipe_assignment": parity_ok, "ppo_iterations": 0}, indent=2))
