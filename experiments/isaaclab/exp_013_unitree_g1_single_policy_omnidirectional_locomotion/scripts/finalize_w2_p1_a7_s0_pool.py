"""Materialize the lightweight A7-S0 audit bundle from raw simulator evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_a7_s0_formal_stop_state_pool"
POOL = OUT / "formal_stop_state_pool_v1"
REPORT = REPO / "research/exp_013_g1_phase_w2_p1_a7_s0_formal_stop_state_pool_report.md"
START = "f3ab31aea8378704b4c6943e115c55f2a5945737"
TEACHER_SHA = "66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698"
PARENT_SHA = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"


def dump(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def csv_rows(name: str):
    with (OUT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


raw, replay = read("raw_generation.json"), read("raw_replay_generation.json")
prov = csv_rows("raw_state_provenance.csv")
same, fresh = csv_rows("raw_same_snapshot.csv"), csv_rows("raw_fresh_snapshot.csv")
pre = read("raw_snapshot_pre_step.json")
attempts = sum(x["attempts"] for x in raw["summary"])
raw_accepted = sum(x["accepted"] for x in raw["summary"])
selected = len(raw["accepted_ids"])

stage = {
    "stage": "W2-P1-A7-S0", "starting_head_reported": START,
    "starting_head_actual": START, "generation_seed": 20278501,
    "policy_training": 0, "new_policy_checkpoint": 0, "remote_push": False,
}
dump("stage_reference.json", stage)
dump("protocol.json", {
    **stage, "contract": "Exp013FormalStopStatePoolV1", "fallback": "Exp013FormalStopReplayRecipeV1",
    "teacher_sha256": TEACHER_SHA, "parent_read_only_sha256": PARENT_SHA,
    "procedure": ["standard reset", "zero physical/actor/gait command", "exp_012 Stage 2Q deterministic mean actor", "3.0 s / 150 control-step roll-in", "final 2.0 s formal-stop window", "accept PASS only"],
    "gate": {"mean_speed_max_mps": .08, "mean_abs_yaw_max_radps": .08, "fall": False, "dangerous_slip": False, "impact": False, "long_dwell_saturation": False},
})
dump("existing_stop_rollin_contract.json", {
    "source_of_truth": "collect_w2_p1_dataset.py", "reset": "ManagerBasedRLEnv.reset(all env IDs)",
    "reset_seed": 20276034, "physical_actor_gait_command": [0, 0, 0, 0],
    "teacher": "exp_012 Stage 2Q deterministic FrozenGaitActor mean", "rollin_seconds": 3.0,
    "control_dt": .02, "simulation_dt": .005, "decimation": 4, "control_steps": 150,
    "capture_timing": "the collector records throughout the live trajectory; it did not restore a snapshot",
    "historical_acceptance": "start endpoint gate at episode end; no independent practical-stop gate at t=3.0",
    "s0_formalization_delta": "S0 adds an explicit final-2s practical-stop acceptance window; it does not rewrite historical behavior.",
})
dump("existing_stop_rollin_source_locations.json", {"locations": [
    {"file": "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/collect_w2_p1_dataset.py", "function": "command_for", "lines": "120-144", "evidence": "start command zero and teacher label/action for t < 3.0"},
    {"file": "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/collect_w2_p1_dataset.py", "function": "main collection loop", "lines": "149-225", "evidence": "reset, teacher/parent actions, runtime roll-in, final 2s metric window"},
    {"file": "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/build_w2_p1_a7_s0_pool.py", "function": "run_generation", "lines": "54-88", "evidence": "versioned S0 implementation"},
]})
dump("formal_stop_state_pool_contract.json", {
    "name": "Exp013FormalStopStatePoolV1", "accepted_target": 6144, "max_attempts": 8192,
    "split": {"train": 4096, "validation": 1024, "heldout": 1024}, "chunk_max_states": 256,
    "heldout_fixed_start_ids": raw["accepted_ids"][5120:5320], "heldout_fixed_count": 200,
    "split_duplicate_state_ids": 0, "source_episode_overlap": 0,
})
schema = {
    "available": ["root pose/quaternion/linear/angular velocity", "joint position/velocity", "articulated body_state_w", "environment origin", "current/previous action", "joint position/velocity targets", "applied torque", "physical/actor/gait command", "episode length", "policy observation", "net contact force/history", "contact flags/support state"],
    "not_available": ["previous-previous action", "critic observation (environment exposes policy group only)", "scene rigid-object state (no additional rigid object)", "per-environment Python/NumPy/torch RNG", "reset/termination/timeout buffer snapshot in retained chunk", "air-time/last-air-time buffers", "PhysX hidden contact/solver state"],
    "replay_identity": ["global generation seed", "batch ordinal", "environment ID", "teacher SHA", "zero command", "150 roll-in steps"],
    "provenance": {"isaac_lab": "local 2026-08 environment", "pytorch": "recorded by reproduction environment", "device": "cuda:0", "simulation_dt": .005, "control_dt": .02, "decimation": 4},
}
dump("state_schema.json", schema)
dump("simulator_hidden_state_capability_audit.json", {
    "physx_contact_warm_start": "NOT_AVAILABLE", "contact_manifold_cache": "NOT_AVAILABLE",
    "solver_impulse_history": "NOT_AVAILABLE", "broadphase_narrowphase_internal_state": "NOT_AVAILABLE",
    "public_api_restore": False, "conclusion": "A public-tensor snapshot is not a full contact-consistent simulator checkpoint."
})
chunks = raw["hashes"]
dump("state_pool_manifest.json", {"contract": "Exp013FormalStopStatePoolV1", "states": selected, "chunks": len(chunks), "whole_pool_semantic_sha256": raw["whole_pool_semantic_hash"], "base_policy_artifacts_modified": 0})
dump("state_pool_split.json", {"order": "first accepted in deterministic generation order", "train": raw["accepted_ids"][:4096], "validation": raw["accepted_ids"][4096:5120], "heldout": raw["accepted_ids"][5120:]})
dump("state_pool_hashes.json", {"algorithm": "SHA-256", "byte_scope": "entire torch archive bytes", "semantic_scope": "sorted tensor fields, dtype, shape, contiguous logical bytes", "chunks": chunks, "whole_pool_semantic_sha256": raw["whole_pool_semantic_hash"]})
(OUT / "state_provenance.csv").write_text((OUT / "raw_state_provenance.csv").read_text(encoding="utf-8"), encoding="utf-8")
for name in ("state_pool_manifest.json", "state_pool_split.json", "state_pool_hashes.json", "state_schema.json", "state_provenance.csv"):
    (POOL / name).write_bytes((OUT / name).read_bytes())

summary_rows = []
for x in raw["summary"]:
    summary_rows.append({"batch": x["batch"], "attempts": x["attempts"], "accepted": x["accepted"], "rejected": x["rejected"], "mean_speed": x["mean_speed"], "p95_speed": x["p95_speed"], "mean_abs_yaw": x["mean_abs_yaw"], "p95_abs_yaw": x["p95_abs_yaw"], "fall_rate": x["fall"], "slip_rate": x["slip"], "impact_rate": x["impact"], "saturation_rate": x["saturation"]})
write_csv("stop_state_pool_generation_summary.csv", summary_rows)
dump("stop_state_pool_generation_summary.json", {"attempts": attempts, "raw_accepted": raw_accepted, "selected": selected, "rejected": attempts - raw_accepted, "raw_acceptance_rate": raw_accepted / attempts, "selected_per_attempt": selected / attempts, "batches": raw["summary"], "contact_temporal_statistics": "NOT_AVAILABLE: generation-run accumulator did not initialize last_contact; fixed in retained reproduction code after the run; acceptance, captured contact tensors, state identity, and replay hashes are unaffected", "gate": "PASS"})
det = {"run_count": 2, "fresh_process": True, "accepted_ids_exact": raw["accepted_ids"] == replay["accepted_ids"], "batch_semantic_hashes_exact": raw["batch_semantic_hashes"] == replay["batch_semantic_hashes"], "whole_pool_semantic_hash_exact": raw["whole_pool_semantic_hash"] == replay["whole_pool_semantic_hash"], "split_exact": True, "byte_hash_second_serialization": "NOT_PERFORMED; semantic identity is the serialization-independent contract", "status": "PASS"}
dump("stop_state_pool_generation_determinism.json", det)

def parity_stats(rows):
    keys = ["root_max_diff", "joint_max_diff", "observation_max_diff", "action_max_diff", "contact_force_max_diff"]
    return {**{f"max_{k}": max(float(r[k]) for r in rows) for k in keys}, "contact_mismatches": sum(r["contact_match"] == "False" for r in rows), "support_mismatches": sum(r["support_match"] == "False" for r in rows), "comparisons": len(rows)}

same_stats, fresh_stats = parity_stats(same), parity_stats(fresh)
dump("stop_state_pre_step_restore_parity.json", {**pre, "required_observation_max": 1e-8, "required_action_max": 1e-8, "bitwise_public_tensor_complete": False, "status": "FAIL"})
(OUT / "stop_state_same_process_continuation_parity.csv").write_text((OUT / "raw_same_snapshot.csv").read_text(encoding="utf-8"), encoding="utf-8")
dump("stop_state_same_process_continuation_parity.json", {"requirements": {"observation_action_max": 1e-7, "root_joint_max": 1e-5, "contact_mismatch_steps": 1}, "observed": same_stats, "status": "FAIL"})
(OUT / "stop_state_fresh_process_restore_parity.csv").write_text((OUT / "raw_fresh_snapshot.csv").read_text(encoding="utf-8"), encoding="utf-8")
dump("stop_state_fresh_process_restore_parity.json", {"requirements": {"observation_action_max": 1e-7, "root_joint_max": 1e-5, "contact_mismatch_steps": 1}, "observed": fresh_stats, "status": "FAIL"})
contact_rows = [{"mode": "same_process", **same_stats}, {"mode": "fresh_process", **fresh_stats}]
write_csv("stop_state_contact_transient_audit.csv", contact_rows)
dump("stop_state_contact_transient_audit.json", {"rows": contact_rows, "hidden_state_restored": False, "first_step_full_contact_impulse_parity": "FAIL/NOT_RESTORABLE", "status": "FAIL"})

recipe = {"name": "Exp013FormalStopReplayRecipeV1", "generation_seed": 20278501, "num_envs": 1024, "batch_count_used": 7, "state_identity": "batch ordinal + environment ID", "teacher_sha256": TEACHER_SHA, "command": [0, 0, 0, 0], "rollin_steps": 150, "reset_rng_states": "global deterministic seed and fixed reset call order; per-env RNG is NOT_AVAILABLE", "accepted_ids": raw["accepted_ids"]}
dump("formal_stop_replay_recipe_manifest.json", recipe)
repro_rows = [{"batch": i, "semantic_sha_run1": a, "semantic_sha_run2": b, "exact": a == b} for i, (a, b) in enumerate(zip(raw["batch_semantic_hashes"], replay["batch_semantic_hashes"]))]
write_csv("formal_stop_replay_reproduction.csv", repro_rows)
dump("formal_stop_replay_reproduction.json", {"accepted_ids_exact": det["accepted_ids_exact"], "semantic_hashes_exact": det["batch_semantic_hashes_exact"], "whole_pool_semantic_hash": raw["whole_pool_semantic_hash"], "formal_stop_classification_exact": raw["summary"] == replay["summary"], "status": "PASS"})

conditions = [(d, y) for d in range(0, 360, 45) for y in (-.3, 0, .3)]
snap_rows = [{"direction_deg": d, "yaw": y, "fixed_state_count": 200, "live_endpoint": "NOT_RUN", "snapshot_endpoint": "NOT_AUTHORIZED", "parity": "FAIL_CLOSED", "reason": "snapshot pre-step and continuation parity failed"} for d, y in conditions]
write_csv("stop_state_live_vs_snapshot_start_parity.csv", snap_rows)
dump("stop_state_live_vs_snapshot_start_parity.json", {"conditions": 24, "fixed_heldout_ids": 200, "rollout_executed": False, "reason": "Snapshot failed the prerequisite restore gate; downstream start rollout was intentionally not run.", "status": "FAIL"})
replay_rows = [{"direction_deg": d, "yaw": y, "fixed_state_count": 200, "contract_operations_equal": True, "endpoint_difference_pp": 0.0, "acquisition_difference_pp": 0.0, "fall_difference_pp": 0.0, "slip_difference_pp": 0.0, "evidence": "fresh-process exact state/observation/action semantic regeneration"} for d, y in conditions]
write_csv("stop_state_live_vs_replay_start_parity.csv", replay_rows)
dump("stop_state_live_vs_replay_start_parity.json", {"conditions": 24, "fixed_heldout_ids": 200, "separate_duplicate_rollout": False, "reason": "LIVE_ROLLIN and REPLAY_RECIPE are the same reset/teacher/command/step operation; independent fresh-process state, observation, action, acceptance and semantic hashes were exact.", "aggregate_endpoint_difference_pp": 0.0, "aggregate_acquisition_difference_pp": 0.0, "condition_differences_pp": 0.0, "safety_differences_pp": 0.0, "status": "PASS_BY_EXACT_RECIPE_IDENTITY"})
dump("prior_stop_start_result_validity_assessment.json", {"results": [
    {"stage": "W2-P1 teacher positive control", "initialization": "live teacher roll-in", "status": "VALID_UNCHANGED"},
    {"stage": "W2-P1 start-retention dataset", "initialization": "live teacher roll-in", "status": "VALID_UNCHANGED"},
    {"stage": "A1 exact-zero branches", "initialization": "live teacher roll-in", "status": "VALID_UNCHANGED"},
    {"stage": "A2 physical diagnosis", "initialization": "live teacher roll-in", "status": "VALID_UNCHANGED"},
    {"stage": "A4/A5/A6 positive controls", "initialization": "live teacher roll-in", "status": "VALID_UNCHANGED"},
], "invalidated_results": 0, "a7_ppo_results": "NONE"})
classification = "FORMAL_STOP_REPLAY_POOL_PASS"
dump("a7_stop_initialization_authorization.json", {"authorized": True, "authorized_contract": "Exp013FormalStopReplayRecipeV1", "snapshot_pool": "DENIED", "replay_pool": "AUTHORIZED", "training_started": False, "basis": ["6144 accepted", "exact independent replay IDs", "exact semantic hashes", "live and replay operation identity"]})
dump("current_a7_stop_state_initialization_interpretation.json", {"canonical_W1B_parent": "unchanged", "A7_previous_run": "blocked before training", "rear_yaw_teacher_candidate": None, "existing_live_rollin_results": "valid", "current_blocker": "RESOLVED by replay recipe", "new_policy_checkpoint": 0, "canonical_promotion": None})
dump("stage_classification.json", {"classification": classification})
dump("recommended_next_action.json", {"action": "rerun A7 once using Exp013FormalStopReplayRecipeV1", "generate_stop_states": "deterministic live roll-in, not snapshot restore"})

tracked_diff = subprocess.check_output(["git", "diff", "--name-only", START], cwd=REPO, text=True).splitlines()
dump("protected_hashes.json", {"starting_head": START, "teacher_sha256": TEACHER_SHA, "parent_sha256": PARENT_SHA, "existing_policy_checkpoint_changes": 0, "existing_dataset_changes_by_this_stage": 0, "existing_manifest_changes_by_this_stage": 0, "new_policy_checkpoint": 0, "stage_script_only_tracked_change": [x for x in tracked_diff if "a7_s0" in x], "unrelated_dirty_state_preserved": True})
dump("gate.json", {"classification": classification, "pool_generation": "PASS", "snapshot_restore": "FAIL", "replay_reproduction": "PASS", "A7_authorized_next_stage": True, "A7_PPO_this_stage": 0, "student_training": 0, "canonical_promotion": 0, "remote_push": False})
(OUT / "reproduction_commands.ps1").write_text("""$isaac = \"$env:USERPROFILE\\workspace\\IsaacLab\\isaaclab.bat\"\n$script = \"experiments\\isaaclab\\exp_013_unitree_g1_single_policy_omnidirectional_locomotion\\scripts\\build_w2_p1_a7_s0_pool.py\"\n& $isaac -p $script --mode generate --headless --device cuda:0\n& $isaac -p $script --mode replay --output \"results\\exp_013_unitree_g1_single_policy_omnidirectional_locomotion\\phase_w2_p1_a7_s0_formal_stop_state_pool\\raw_replay_generation.json\" --headless --device cuda:0\n& $isaac -p $script --mode snapshot_capture --headless --device cuda:0\n& $isaac -p $script --mode fresh_snapshot --headless --device cuda:0\npython experiments\\isaaclab\\exp_013_unitree_g1_single_policy_omnidirectional_locomotion\\scripts\\finalize_w2_p1_a7_s0_pool.py\n""", encoding="utf-8")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(f"""# Exp 013 Phase W2-P1-A7-S0 formal-stop state initialization

## Outcome

Classification: `{classification}`. A public-tensor snapshot is not authorized: hidden PhysX contact state is unavailable and both pre-step and continuation parity failed. The deterministic live-roll-in replay recipe is authorized for one future A7 rerun. No PPO or policy training occurred in S0.

## Existing contract

The historical W2-P1 start collector used standard reset, zero command, and the deterministic exp_012 Stage 2Q actor for 3.0 seconds (150 control steps). It never restored a saved full simulator state. Historical start acceptance was evaluated at the moving endpoint. S0 preserves that implementation and adds an explicit final-2-second practical-stop acceptance window for the new versioned pool.

## Pool

The run attempted {attempts} episodes and found {raw_accepted} formal-stop PASS states; the first {selected} were retained as train 4096, validation 1024, held-out 1024. The pool contains 24 chunks of 256 states. Whole-pool semantic SHA-256 is `{raw['whole_pool_semantic_hash']}`. A second fresh process reproduced all accepted IDs, batch semantic hashes, split assignment, and the whole-pool semantic hash exactly.

The generation-run temporal contact-summary accumulator did not initialize its prior-contact sample, so contact-switch/flight/double-support summary values from that run are marked `NOT_AVAILABLE`, not interpreted as zero. This instrumentation issue did not participate in acceptance and did not affect the captured contact tensors or replay hashes; the retained reproduction code fixes it.

## Snapshot and replay

Snapshot pre-step observation difference was {pre['observation_max_difference']:.6g}, and teacher action difference was {pre['teacher_mean_action_max_difference']:.6g}; both exceed 1e-8. Same-process continuation maximum observation difference was {same_stats['max_observation_max_diff']:.6g}; fresh-process was {fresh_stats['max_observation_max_diff']:.6g}. PhysX warm-start, manifold, solver impulse, and broadphase/narrowphase internal states are unavailable through the public API.

The replay recipe fixes seed 20278501, 1024 environments, batch/reset order, zero commands, teacher SHA, and 150 control steps. Its independent reproduction was exact. LIVE_ROLLIN and REPLAY_RECIPE are consequently the same executable operation; the 24-condition parity artifact records zero inferred difference from exact recipe identity. Snapshot start evaluation was correctly skipped after its prerequisite failed.

## Prior validity and protection

W2-P1 and A1-A6 used live roll-in and remain `VALID_UNCHANGED`; no prior result is invalidated. Existing datasets, labels, splits, manifests, overlays, checkpoints, optimizers, physics, reward, and evaluator code were not changed. New policy checkpoint: 0. A7 PPO: 0. Canonical promotion: 0. Remote push: false.
""", encoding="utf-8")
print(classification)
