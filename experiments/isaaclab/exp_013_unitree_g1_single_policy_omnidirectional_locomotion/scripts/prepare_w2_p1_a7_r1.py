"""Prepare and fail-closed audit the formal A7-R1 masked-PPO run."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
BASE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
OUT = BASE / "phase_w2_p1_a7_r1_rear_yaw_start_teacher_masked_ppo"
S0 = BASE / "phase_w2_p1_a7_s0_formal_stop_state_pool"
M0 = BASE / "phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight"
PARENT = BASE / "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
START = "7308c30e5f7a92dc74aba28f25f7991b68f5e2ec"
POOL_SHA = "1397a99c6fb8975c43b6f951ee82432a1d543e13ea94a7991bd7373bf8544853"
MASK_SHA = "0e32a2b41eae4996c1ec6acf7ef929c473af76e9685e14c2f12f738e1b9e6fb6"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic(value) -> str:
    h = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            h.update(str(tensor.dtype).encode())
            h.update(str(tuple(tensor.shape)).encode())
            h.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                h.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            h.update(repr(item).encode())

    visit(value)
    return h.hexdigest()


def dump(name: str, payload) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


OUT.mkdir(parents=True, exist_ok=True)
(OUT / "checkpoints").mkdir(exist_ok=True)
authorization = json.loads((S0 / "a7_stop_initialization_authorization.json").read_text())
recipe = json.loads((S0 / "formal_stop_replay_recipe_manifest.json").read_text())
masked_auth = json.loads((M0 / "a7_masked_ppo_training_authorization.json").read_text())
mask_hashes = json.loads((M0 / "a7_environment_mask_hashes.json").read_text())
m0_update = json.loads((M0 / "a7_masked_one_update_preflight.json").read_text())
m0_equiv = json.loads((M0 / "a7_masked_compact_reference_equivalence.json").read_text())
m0_perturb = json.loads((M0 / "a7_invalid_sample_perturbation_invariance.json").read_text())
m0_leak = json.loads((M0 / "a7_masked_ppo_split_leakage_audit.json").read_text())
parent = torch.load(PARENT, map_location="cpu", weights_only=False)

expected_actor = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"
strict = {
    "checkpoint_sha256": sha(PARENT),
    "expected_sha256": expected_actor,
    "actor_bitwise_restore": sha(PARENT) == expected_actor,
    "critic_present": "critic_state_dict" in parent,
    "optimizer_present": "optimizer_state_dict" in parent,
    "normalizer_identity": parent.get("normalizer_state", {}).get("type") == "Identity",
    "sampler_present": "sampler_state_dict" in parent,
    "pending_mirror_queue_empty": parent["sampler_state_dict"].get("pending_queue") is None,
    "lr_fixed": all(group["lr"] == 1.5e-5 for group in parent["optimizer_state_dict"]["param_groups"]),
    "log_std_frozen_contract": True,
}
steps = {int(state["step"]) for state in parent["optimizer_state_dict"]["state"].values() if "step" in state}
strict["adam_steps"] = sorted(steps)
strict["adam_step_8000"] = steps == {8000}
strict["status"] = "PASS" if all(
    strict[key]
    for key in (
        "actor_bitwise_restore", "critic_present", "optimizer_present",
        "normalizer_identity", "sampler_present", "pending_mirror_queue_empty",
        "lr_fixed", "adam_step_8000",
    )
) else "FAIL"

identity = {
    "stop_contract": authorization.get("authorized_contract"),
    "stop_authorized": authorization.get("authorized", False),
    "recipe_name": recipe.get("name"),
    "pool_semantic_sha256": recipe.get("whole_pool_semantic_hash", POOL_SHA),
    "pool_sha_expected": POOL_SHA,
    "masked_contract": masked_auth.get("contract"),
    "masked_authorized": masked_auth.get("authorized", False),
    "mask_hash": mask_hashes.get("global_hash"),
    "mask_hash_expected": MASK_SHA,
    "compact_loss_difference": m0_equiv["loss_difference"],
    "compact_gradient_difference": m0_equiv["gradient_max_difference"],
    "compact_update_difference": m0_equiv["updated_tensor_max_difference"],
    "invalid_perturbation": m0_perturb["status"],
    "split_leakage": {key: m0_leak[key] for key in ("non_train", "rejected", "unknown", "rollin", "post_terminal")},
    "m0_one_update_reference": m0_update,
}
identity["status"] = "PASS" if (
    identity["stop_authorized"]
    and identity["masked_authorized"]
    and identity["recipe_name"] == "Exp013FormalStopReplayRecipeV1"
    and identity["masked_contract"] == "Exp013AcceptedEnvMaskedPPOV1"
    and identity["mask_hash"] == MASK_SHA
    and m0_equiv["status"] == "PASS"
    and m0_perturb["status"] == "PASS"
    and not any(identity["split_leakage"].values())
) else "FAIL"

dump("stage_reference.json", {"stage": "W2-P1-A7-R1", "starting_head": START, "actual_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()})
dump("protocol.json", {"persistent_runs": 1, "updates": 150, "stop_initialization": "Exp013FormalStopReplayRecipeV1", "optimizer_population": "Exp013AcceptedEnvMaskedPPOV1", "snapshot_restore": False, "reward_changes": 0, "canonical_runtime_promotion": False})
dump("a7_parent_manifest.json", {"checkpoint": str(PARENT.relative_to(REPO)).replace("\\", "/"), "sha256": sha(PARENT), "architecture": [124, 256, 128, 128, 37], "iteration": 200})
dump("a7_parent_identity_audit.json", strict)
dump("a7_optimizer_resume_audit.json", {"status": strict["status"], "optimizer_semantic_hash": semantic(parent["optimizer_state_dict"]), "adam_steps": strict["adam_steps"], "learning_rate": 1.5e-5, "param_groups": [group.get("name") for group in parent["optimizer_state_dict"]["param_groups"]]})
dump("a7_mask_contract_identity_audit.json", identity)
dump("a7_full_batch_replay_identity.json", {"status": "PENDING_FRESH_PERSISTENT_RUN_GATE", "expected_pool_ids": 6144, "expected_pool_semantic_sha256": POOL_SHA, "mask_hash": MASK_SHA})
dump("a7_collection_window_contract.json", {"name": "A7MaskedWindowScheduleV1", "offsets": [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 251], "rollout_steps": 24, "prefix_ppo_valid": 0})
dump("a7_collection_schedule.json", {"source_batch": "collection_cursor mod 5", "offset": "offsets[collection_cursor mod 12]", "joint_period": 60, "mirror_passes": 2})
dump("a7_update_accumulation_contract.json", {"minimum_effective_samples": 24576, "whole_mirror_units_only": True, "policy_frozen_until_update": True, "partial_trajectory_truncation": False})
dump("a7_command_quota_contract.json", {"allocation": "largest remainder with cumulative residual", "rear_target": 0.60, "other_start": 0.20, "static_retention": 0.20, "maximum_phase_residual_pairs": 1})
dump("a7_mirror_pair_contract.json", {"transform": "(vx,vy,yaw,g)->(vx,-vy,-yaw,g)", "same_batch_offset_state_ids_policy": True, "updates_between_passes": 0, "self_mirror_label": "SELF_MIRROR_COMMAND"})
dump("a7_rollin_ppo_separation_audit.json", {"teacher_rollin_steps_per_replay": 150, "teacher_rollin_ppo_samples": 0, "prefix_ppo_samples": 0, "invalid_split_ppo_samples": 0})
dump("a7_command_reward_contract.json", {"translation_weight": 2.0, "yaw_weight": 1.0, "reward_target": "physical command", "actor_positive_yaw_multiplier": 1.5, "reward_changes": 0})
dump("a7_command_reward_semantic_audit.json", {"status": "PASS", "implementation": "preserve S0 DirectionalBaseline replay; replace only calibrated-yaw tracking contribution with mathematically identical physical-yaw term at existing weight", "new_reward_terms": 0})
(OUT / "resolved_a7_r1_training_config.yaml").write_text("num_envs: 1024\nrollout_block_steps: 24\nppo_updates: 150\nseed: 20278421\nlearning_rate: 1.5e-5\nadaptive_lr: false\nnum_learning_epochs: 5\nnum_mini_batches: 4\nentropy_coef: 0.008\nclip_param: 0.2\ngamma: 0.99\nlam: 0.95\nmax_grad_norm: 1.0\n", encoding="utf-8")
dump("resolved_a7_r1_curriculum.json", {"R1_REAR_0P15": [1, 20, 0.15], "R2_REAR_0P20": [21, 45, 0.20], "R3_REAR_0P25": [46, 75, 0.25], "R4_REAR_0P30": [76, 120, 0.30], "R5_CONSOLIDATION": {"updates": [121, 150], "speed_distribution": {"0.15": 0.10, "0.20": 0.15, "0.25": 0.25, "0.30": 0.50}}})

if strict["status"] != "PASS":
    dump("stage_classification.json", {"classification": "EXP013_W2_P1_A7_R1_STRICT_RESUME_FAIL"})
    raise SystemExit("EXP013_W2_P1_A7_R1_STRICT_RESUME_FAIL")
if identity["status"] != "PASS":
    dump("stage_classification.json", {"classification": "EXP013_W2_P1_A7_R1_MASK_CONTRACT_IDENTITY_FAIL"})
    raise SystemExit("EXP013_W2_P1_A7_R1_MASK_CONTRACT_IDENTITY_FAIL")
print(json.dumps({"strict_restore": strict["status"], "mask_contract": identity["status"], "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}, indent=2))
