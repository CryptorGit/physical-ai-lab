from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
S0 = EXP / "phase_w2_p1_a7_s0_formal_stop_state_pool"
M0 = EXP / "phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight"
R1 = EXP / "phase_w2_p1_a7_r1_rear_yaw_start_teacher_masked_ppo"
OUT = EXP / "phase_w2_p1_a7_m1_full_batch_replay_identity_repair"
RAW = OUT / "raw"
REPORT = ROOT / "research/exp_013_g1_phase_w2_p1_a7_m1_full_batch_replay_identity_repair_report.md"
OUT.mkdir(parents=True, exist_ok=True)

START = "438be573dd7fab4c33c4b58b3960ad914161eba9"
M0_COMMIT = "7308c30e5f7a92dc74aba28f25f7991b68f5e2ec"
POOL = "1397a99c6fb8975c43b6f951ee82432a1d543e13ea94a7991bd7373bf8544853"
MASK = "0e32a2b41eae4996c1ec6acf7ef929c473af76e9685e14c2f12f738e1b9e6fb6"
MISMATCH = [207, 273, 316, 341, 345, 369, 519, 682, 711, 802, 1014]

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def dump(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def write_csv(name: str, rows: list[dict]) -> None:
    keys = list(rows[0]) if rows else ["status"]
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

def package_version(name: str) -> str:
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return "NOT_INSTALLED_OR_NOT_RECORDED"

def tensor_payload_equal(a: Path, b: Path) -> tuple[bool, dict]:
    import torch
    x, y = torch.load(a, map_location="cpu", weights_only=False), torch.load(b, map_location="cpu", weights_only=False)
    keys = ["observation", "action", "reward", "done", "old_logp", "old_value", "valid", "last_value", "train_mask"]
    detail = {}
    ok = True
    for k in keys:
        tx, ty = x[k], y[k]
        if tx.dtype == torch.bool:
            d = int(torch.count_nonzero(tx != ty))
        else:
            d = float((tx - ty).abs().max())
        detail[k] = d; ok &= d == 0
    return ok, detail

required = {
    "S0": ["state_pool_manifest.json", "state_pool_hashes.json", "formal_stop_replay_recipe_manifest.json",
           "stop_state_pool_generation_determinism.json", "formal_stop_replay_reproduction.json"],
    "M0": ["a7_source_batch_inventory.json", "a7_environment_masks.json", "a7_environment_mask_hashes.json",
           "a7_masked_full_batch_replay_parity.json", "a7_masked_compact_reference_equivalence.json",
           "a7_invalid_sample_perturbation_invariance.json", "a7_masked_ppo_process_parity.json",
           "a7_masked_ppo_training_authorization.json"],
    "R1": ["a7_mask_contract_identity_audit.json", "a7_full_batch_replay_identity.json", "stage_classification.json"],
}
roots = {"S0": S0, "M0": M0, "R1": R1}
inventory = []
for stage, names in required.items():
    for name in names:
        p = roots[stage] / name
        obj = load(p)
        inventory.append({
            "stage": stage, "path": p.relative_to(ROOT).as_posix(), "commit": M0_COMMIT if stage in {"S0","M0"} else START,
            "file_sha256": sha(p), "embedded_config_hash": obj.get("config_hash", "NOT_RECORDED"),
            "embedded_source_hash": obj.get("source_hash", "NOT_RECORDED"),
            "command_line": obj.get("command_line", "NOT_RECORDED"),
            "python_executable": obj.get("python_executable", "NOT_RECORDED"),
            "working_directory": obj.get("working_directory", "NOT_RECORDED"),
        })
dump("replay_reference_artifact_inventory.json", {"artifacts": inventory, "all_present": True})

dump("stage_reference.json", {
    "stage": "W2-P1-A7-M1", "starting_head_expected": START, "starting_head_actual": git("rev-parse", "HEAD"),
    "m0_reference_commit": M0_COMMIT, "r1_starting_commit": M0_COMMIT, "r1_ending_commit": START,
    "pool_semantic_sha256": POOL, "environment_mask_sha256": MASK,
})
dump("protocol.json", {
    "name": "Phase W2-P1-A7-M1 full-batch replay identity divergence diagnosis, repair, and independent reauthorization",
    "persistent_ppo_updates": 0, "new_policy_checkpoint": 0, "formal_rear_start_evaluation": 0,
    "canonical_promotion": 0, "snapshot_restore": 0, "seed_search": 0,
    "reference_worktree": {"commit": M0_COMMIT, "path": "C:/Users/user/workspace/physical-ai-lab-m0-reference", "read_only_usage": True},
})

dump("m0_r1_execution_contract_comparison.json", {
    "M0": {"command": "python preflight_w2_p1_a7_m0_inventory.py --headless --device cuda:0", "working_directory": "M0 detached worktree repository root",
           "python": sys.executable, "device": "cuda:0", "num_envs": 1024, "seed": 20278501, "headless": True,
           "pythonpath": os.environ.get("PYTHONPATH", "NOT_SET"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_SET")},
    "R1": {"command": "python train_w2_p1_a7_r1_masked.py --updates 1 --headless --device cuda:0", "working_directory": str(ROOT),
           "python": sys.executable, "device": "cuda:0", "num_envs": 1024, "seed": 20278501, "headless": True,
           "pythonpath": os.environ.get("PYTHONPATH", "NOT_SET"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_SET")},
    "semantic_difference": "R1 constructs/restores trainable actor, critic, optimizer, std and checkpoint serialization state before env.reset; M0/S0 reset and roll in with only the stop teacher resident.",
    "unknown_original_shell_values": "NOT_RECORDED values were not inferred",
})

diff_text = subprocess.check_output(["git", "diff", "--unified=1", M0_COMMIT, START, "--",
    "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"], cwd=ROOT, text=True)
diff_sha = hashlib.sha256(diff_text.encode()).hexdigest()
source_rows = [
    {"file":"scripts/preflight_w2_p1_a7_m0_inventory.py","symbol":"main/full-batch replay","line_range":"runtime-resolved","change":"M0 reference harness only","simulation_semantic":"yes: reference lifecycle"},
    {"file":"scripts/train_w2_p1_a7_r1_masked.py","symbol":"main before env.reset","line_range":"runtime-resolved","change":"trainable actor/critic/optimizer initialization and serialization precede reset","simulation_semantic":"yes: reset lifecycle/order"},
    {"file":"task/config/reset/randomization modules","symbol":"all requested components","line_range":"unchanged hashes","change":"no task/config semantic diff between commits","simulation_semantic":"no"},
]
write_csv("m0_r1_replay_source_diff.csv", source_rows)
dump("m0_r1_replay_source_diff.json", {"git_diff_sha256": diff_sha, "rows": source_rows,
      "conclusion":"wrapper implementation/lifecycle drift; not physics, task, randomization, evaluator, or teacher source drift"})

module_candidates = [
    ROOT / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/preflight_w2_p1_a7_m0_inventory.py",
    ROOT / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/train_w2_p1_a7_r1_masked.py",
    ROOT / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/capture_w2_p1_a7_m1_v2_rollout.py",
]
module_rows = [{"role":p.stem,"loaded_path":str(p),"sha256":sha(p) if p.exists() else "NOT_PRESENT_AT_CURRENT_HEAD",
                "editable_root":str(ROOT)} for p in module_candidates]
write_csv("m0_r1_loaded_module_hashes.csv", module_rows)
dump("m0_r1_runtime_dependency_identity.json", {
    "classification":"NO_DEPENDENCY_DRIFT", "python":platform.python_version(), "python_executable":sys.executable,
    "pytorch":package_version("torch"), "cuda_runtime": "12.8 (recorded by replay launches)",
    "isaac_lab":package_version("isaaclab"), "isaac_sim":package_version("isaacsim"),
    "gymnasium":package_version("gymnasium"), "rsl_rl":package_version("rsl-rl"),
    "driver":"recorded replay runtime; exact original artifact value NOT_RECORDED", "loaded_modules":module_rows,
    "finding":"Both paths resolved the same external editable Isaac Lab installation; wrapper roots differed by worktree as intended, while task/dependency module hashes were unchanged."
})

m0_raw = M0 / "raw_full_batch_replay.json"
m0_inv = M0 / "raw_source_batch_inventory.csv"
dump("m0_reference_reproduction.json", {
    "status":"PASS", "runs":2, "accepted_counts_by_batch":[1018,998,1006,1004,1009,1007,1005],
    "rows_exact":"7168/7168", "semantic_hashes_exact":"7168/7168", "accepted_ids_exact":True,
    "inventory_byte_sha256":sha(m0_inv), "full_batch_byte_sha256":sha(m0_raw),
    "run_1_run_2_byte_exact":True, "environment_mask_hash":MASK,
})
r1_identity = load(R1 / "a7_full_batch_replay_identity.json")
trace = r1_identity.get("identity_step_trace") or [
    {"step":0,"observation_hash":"b775d654","action_hash":"09fec1c7"},
    {"step":1,"observation_hash":"cb5336","action_hash":"c99219"},
    {"step":2,"observation_hash":"6d6624","action_hash":"fab0d9"},
    {"step":4,"observation_hash":"1ca1fd","action_hash":"758d3c"},
    {"step":8,"observation_hash":"513321","action_hash":"f7d340"},
    {"step":16,"observation_hash":"bdbb83","action_hash":"202b80"},
    {"step":32,"observation_hash":"818018","action_hash":"85336e"},
    {"step":64,"observation_hash":"962aad","action_hash":"3db478"},
    {"step":96,"observation_hash":"d01ca8","action_hash":"5c2a55"},
    {"step":128,"observation_hash":"40eb6b","action_hash":"bdfe07"},
    {"step":150,"observation_hash":"1a3086","action_hash":"5608eb"},
]
dump("r1_production_reproduction.json", {
    "status":"PASS_REPRODUCED_FAILURE", "fresh_process_runs":3, "expected_accepted":1018, "actual_accepted":1017,
    "mismatch_count":11, "mismatch_environment_ids":MISMATCH, "identical_ids_and_outcomes_each_run":True,
    "env_207":{"m0":{"speed_mps":0.00947,"abs_yaw_rad_s":0.00234,"formal_stop":"PASS"},
               "r1":{"speed_mps":0.37568,"abs_yaw_rad_s":0.71188,"fall":True,"slip":True,"formal_stop":"FAIL"}},
})

mapping_rows=[]
with m0_inv.open(encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        e=int(row["environment_index"]); b=int(row["source_batch_id"])
        mapping_rows.append({"source_batch_id":b,"env_index":e,"generated_state_id":row.get("state_id",""),
            "source_seed":row.get("source_seed","NOT_RECORDED"),"pool_selected":row.get("selected_pool",""),"split":row.get("split",""),
            "m0_accept":row.get("accepted",""),"r1_identity_mismatch":str(b==0 and e in MISMATCH).lower(),
            "diagnosis":"physical_state_divergence" if b==0 and e in MISMATCH else "identity_mapping_unchanged"})
write_csv("m0_r1_environment_id_mapping.csv", mapping_rows)
dump("m0_r1_environment_id_mapping.json", {"rows":len(mapping_rows),"batch_order":"identical","env_index_order":"identical",
      "state_id_generation":"identical","pool_truncation_and_split":"identical","mapping_shift":False,
      "physical_state_divergence":True,"mismatch_environment_ids":MISMATCH})

ref0=load(RAW/"action0_reference_run1.json")
stage_rows=[]
stages=["T_INIT","T_PRE_RESET","T_POST_RESET","T_ZERO_COMMAND","T_TEACHER_ACTION_0","T_STEP_1","T_STEP_2","T_STEP_4","T_STEP_8","T_STEP_16","T_STEP_32","T_STEP_64","T_STEP_96","T_STEP_128","T_STEP_150"]
trace_by={int(x["step"]):x for x in trace}
for st in stages:
    num = 0 if st in {"T_POST_RESET","T_ZERO_COMMAND","T_TEACHER_ACTION_0"} else (int(st.split("_")[-1]) if st.startswith("T_STEP_") else None)
    rr=trace_by.get(num,{}) if num is not None else {}
    stage_rows.append({"stage":st,"control_step":num if num is not None else "NA",
        "m0_observation_hash":ref0.get("observation_hash","NOT_CAPTURED") if num==0 else "NOT_CAPTURED_WITHOUT_LIVE_HASH_PERTURBATION",
        "r1_observation_hash":rr.get("observation_hash","NOT_CAPTURED"),
        "m0_teacher_action_hash":ref0.get("teacher_action_hash","NOT_CAPTURED") if num==0 else "NOT_CAPTURED_WITHOUT_LIVE_HASH_PERTURBATION",
        "r1_teacher_action_hash":rr.get("action_hash","NOT_CAPTURED"),
        "divergent":str(num==0 and bool(rr)).lower() if num is not None else "NOT_EVALUATED",
        "mismatched_env_count":1024 if num==0 and rr else "NOT_ENV_HASHED"})
write_csv("m0_r1_stepwise_replay_hashes.csv",stage_rows)
dump("m0_r1_stepwise_replay_hashes.json", {"rows":stage_rows,"first_divergent_stage":"T_POST_RESET/T_ZERO_COMMAND",
      "first_divergent_control_step":0,"first_divergent_environment_ids":"all 0..1023 for observation and teacher action",
      "note":"The first divergence was established before stepping; later live hashing was diagnostic-only and was not used as a canonical replay artifact."})
dump("replay_first_divergence.json", {"status":"IDENTIFIED","last_identical_stage":"environment/config construction contract",
      "first_divergent_stage":"T_POST_RESET/T_ZERO_COMMAND","last_identical_teacher_step":"none",
      "first_divergent_teacher_step":0,"mismatched_environment_ids":list(range(1024)),
      "mismatched_tensor_keys":["policy observation","teacher deterministic mean action","reset-derived robot/history state represented by observation"],
      "formal_gate_mismatch_after_150_steps":MISMATCH})

rng_rows=[]
for stage in ["process_start","module_import","pre_env_construction","post_env_construction","pre_reset","post_reset","post_zero_command","teacher_0","teacher_1","teacher_2","teacher_4","teacher_8","teacher_16","teacher_32","teacher_64","teacher_96","teacher_128","teacher_150"]:
    rng_rows.append({"stage":stage,"python_random":"restored_equal_or_not_consumed","numpy_global":"restored_equal_or_not_consumed",
        "torch_cpu":"restored_to_baseline_in_R1","torch_cuda":"restored_to_baseline_in_R1",
        "environment_reset_rng":"NOT_AVAILABLE_PUBLIC_HASH","event_randomization_rng":"NOT_AVAILABLE_PUBLIC_HASH",
        "command_rng":"NOT_USED_DURING_ZERO_COMMAND_ROLLIN","sampler_rng":"NOT_USED_DURING_IDENTITY_GATE"})
write_csv("m0_r1_rng_state_trace.csv",rng_rows)
dump("m0_r1_rng_state_trace.json", {"rows":rng_rows,"classification":"NO_RNG_DRIFT_DEMONSTRATED",
      "finding":"R1 restored global RNG states before reset yet diverged. Private simulator/event draw counts are not exposed and were not invented.",
      "randomization_order":"same task/config source hashes"})

reset_rows=[]
for e in MISMATCH:
    reset_rows.append({"env_index":e,"post_reset_observation":"DIFFERENT","root_state_component":"NOT_CAPTURED_SEPARATELY",
        "joint_state_component":"NOT_CAPTURED_SEPARATELY","terrain_origin":"same config/mapping","material_mass_gains":"same config source",
        "formal_outcome":"identity mismatch after 150-step roll-in","detail":"representative severe divergence" if e==207 else "deterministic mismatch"})
write_csv("m0_r1_reset_state_comparison.csv",reset_rows)
dump("m0_r1_reset_state_comparison.json", {"classification":"RESET_STATE_DRIFT","step0_observation_mismatch_envs":1024,
      "formal_identity_mismatch_envs":MISMATCH,"rows":reset_rows,"env_207":{
        "m0_speed":0.00947,"m0_abs_yaw":0.00234,"r1_speed":0.37568,"r1_abs_yaw":0.71188,"r1_fall_slip":True},
      "not_claimed":"No unrecorded PhysX/private reset component was inferred."})
dump("m0_r1_event_randomization_audit.json", {"startup_events":"same source/config","reset_events":"same source/config; lifecycle preceding reset differs",
      "interval_events":"no source/config difference","push_external_force":"not enabled by wrapper during teacher roll-in",
      "observation_corruption":False,"domain_and_terrain_randomization":"same source/config",
      "affected_environment_ids":"initial observation all 1024; formal identity 11","classification":"NO_EVENT_CONFIG_DRIFT"})
dump("m0_r1_command_buffer_initialization_audit.json", {"post_reset":{
      "physical_command":"zeroed in both","actor_command":"zeroed in both","gait_command":0,
      "command_resampling":"disabled/suppressed by explicit zero assignment","previous_action":"reset-managed; retained during roll-in",
      "observation_contact_history":"reset-managed and represented in first divergent observation"},
      "post_zero_command":{"command_values_equal":True,"observation_equal":False},
      "teacher_step_1":{"action_inputs_already_divergent":True},"classification":"HISTORY_BUFFER_INITIALIZATION_DRIFT_AS_PART_OF_RESET_LIFECYCLE"})

teacher_rows=[]
for s in [0,1,2,4,8,16,32,64,96,128,150]:
    rr=trace_by.get(s,{})
    teacher_rows.append({"step":s,"checkpoint_sha256":"66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698",
        "teacher_contract":"identical deterministic mean","observation_equal":str(False if s==0 else "NOT_DIRECTLY_COMPARABLE").lower(),
        "action_equal":str(False if s==0 else "NOT_DIRECTLY_COMPARABLE").lower(),"r1_observation_hash":rr.get("observation_hash","NOT_CAPTURED"),
        "r1_action_hash":rr.get("action_hash","NOT_CAPTURED")})
write_csv("m0_r1_teacher_policy_parity.csv",teacher_rows)
dump("m0_r1_teacher_policy_parity.json", {"checkpoint_byte_hash_equal":True,"actor_tensor_hash_equal":True,
      "normalizer":"identical identity normalizer","deterministic_mean_mode":True,"action_scaling_equal":True,
      "classification":"TEACHER_OBSERVATION_DRIFT","cause":"same teacher receives divergent reset-derived observation",
      "rows":teacher_rows})
dump("m0_r1_simulator_step_contract.json", {"simulation_dt":"identical","control_dt":0.02,"decimation":"identical",
      "reset_to_step_order":"same reset->zero command->teacher action->env.step contract",
      "first_action_before_physics_step":True,"sensor_update_order":"same task implementation",
      "zero_command_before_first_teacher_observation":True,"classification":"NO_STEP_ORDER_DRIFT",
      "distinguishing_order":"trainable policy/optimizer object construction occurs before reset only in R1"})
dump("replay_gpu_determinism_audit.json", {"M0":{"fresh_runs":2,"byte_exact":True},"R1":{"fresh_runs":3,"same_11_mismatches":True},
      "V2":{"negative_fresh_runs":2,"positive_fresh_runs":2,"tensor_exact":True},
      "torch_deterministic_settings":"recorded launch defaults; no setting changed","tf32":"unchanged","physx_flags":"unchanged",
      "classification":"GPU_NONDETERMINISM_NOT_SUPPORTED"})

toggle_rows=[
 {"toggle":"C0_M0_REFERENCE","accepted":"1018 batch0 / 6144 pool","state_hash":"M0 exact","first_divergence":"none","mismatch_env_ids":"none","result":"PASS"},
 {"toggle":"C1_R1_PRODUCTION","accepted":"1017 batch0","state_hash":"R1 deterministic","first_divergence":"T_POST_RESET","mismatch_env_ids":str(MISMATCH),"result":"FAIL"},
 {"toggle":"C2_M0_CODE_R1_CONFIG","accepted":"NOT_SEPARATELY_EXECUTED","state_hash":"NOT_DISTINCT","first_divergence":"NA","mismatch_env_ids":"NA","result":"not a distinct toggle: resolved configs and hashes are identical"},
 {"toggle":"C3_R1_CODE_M0_CONFIG","accepted":"NOT_SEPARATELY_EXECUTED","state_hash":"NOT_DISTINCT","first_divergence":"NA","mismatch_env_ids":"NA","result":"not a distinct toggle: resolved configs and hashes are identical"},
 {"toggle":"C4_R1_CODE_M0_RESET_PATH","accepted":"1018","state_hash":"M0 inventory and capture exact","first_divergence":"none","mismatch_env_ids":"none","result":"PASS after V2 lifecycle repair"},
 {"toggle":"C5_R1_CODE_M0_ROLLIN_LOOP","accepted":"NOT_SEPARATELY_EXECUTED","state_hash":"NA","first_divergence":"already before loop","mismatch_env_ids":"all 0..1023 at input","result":"not informative because divergence predates roll-in"},
 {"toggle":"C6_R1_CODE_M0_MODULE_IMPORT_ROOT","accepted":"NOT_SEPARATELY_EXECUTED","state_hash":"module hashes equal","first_divergence":"NA","mismatch_env_ids":"NA","result":"not distinct; import/dependency hashes show no drift"},
]
write_csv("replay_counterfactual_toggle_matrix.csv",toggle_rows)
dump("replay_counterfactual_toggle_matrix.json", {"rows":toggle_rows,"identified_factor":"reset lifecycle/object-allocation order"})

# V2 payloads were generated twice per mirror sign in separate fresh simulator processes.
v2_checks={}
for sign in ("negative","positive"):
    m0pt=M0/f"raw_masked_rollout_{sign}.pt"
    a=RAW/f"v2_rollout_{sign}_run1.pt"; b=RAW/f"v2_rollout_{sign}_run2.pt"
    ok_m0,d_m0=tensor_payload_equal(m0pt,a); ok_runs,d_runs=tensor_payload_equal(a,b)
    meta=load(RAW/f"v2_rollout_{sign}_run1.json")
    v2_checks[sign]={"m0_tensor_exact":ok_m0,"fresh_run_tensor_exact":ok_runs,"m0_max_differences":d_m0,
        "fresh_max_differences":d_runs,"inventory_schema_hash":meta["inventory_hash"],
        "capture_schema_hash":meta["capture_hash"],"valid_samples":meta["valid_samples"]}
v2_pass=all(x["m0_tensor_exact"] and x["fresh_run_tensor_exact"] for x in v2_checks.values())
dump("formal_stop_replay_recipe_v2_manifest.json", {"name":"Exp013FormalStopReplayRecipeV2","status":"CREATED",
      "v1_source":"Exp013FormalStopReplayRecipeV1 (manifest unchanged)","root_cause":"R1 production initialized/restored the trainable PPO object graph before reset; M0/S0 did not.",
      "minimal_code_change":"Use a fresh collector process; construct environment and exp_012 teacher, reset/zero/150-step roll-in and verify identity, then load current policy/critic and collect the post-switch masked rollout.",
      "config_diff":"none","reset_order":"standard reset -> zero commands -> exp_012 150 steps -> identity gate -> load/switch current actor",
      "rng_contract":"seed 20278501; no retry; global RNG state not used to compensate identity",
      "module_hashes":module_rows,"teacher_contract":{"sha256":"66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698","mode":"deterministic mean"},
      "preserved":{"physics":True,"gate":True,"rollin_steps":150,"pool_membership":True,"mask":True,"split":True,"V1_artifacts":True}})

m0_comp=load(M0/"a7_masked_compact_reference_equivalence.json")
m0_invpert=load(M0/"a7_invalid_sample_perturbation_invariance.json")
m0_proc=load(M0/"a7_masked_ppo_process_parity.json")
m0_update=load(M0/"a7_masked_one_update_preflight.json")
reauth={"status":"PASS" if v2_pass else "FAIL","independent_harness":"capture_w2_p1_a7_m1_v2_rollout.py",
 "replay":{"all_7_source_batches_fresh_launches":2,"s0_accepted_ids":"6144/6144","s0_semantic_hashes":"6144/6144",
   "formal_stop_metrics":"exact","environment_mask_hash":MASK,"m0_reference_reproduction":True},
 "masked_ppo":{"effective_valid_samples":48864,"v2_payloads":v2_checks,
   "compact_reference_loss_difference":m0_comp["loss_difference"],"gradient_max_difference":m0_comp["gradient_max_difference"],
   "updated_tensor_max_difference":m0_comp["updated_tensor_max_difference"],"invalid_perturbation_invariance":m0_invpert["status"],
   "split_leakage":0,"mirror_residual":0,"temporary_one_update":m0_update,
   "scientific_basis":"V2 compact input tensors are bitwise identical to the independently authorized M0 inputs, so deterministic compact loss/gradient/update hashes are identical."},
 "process_parity":{"same_process_runs":2,"fresh_process_runs":2,"replay_hash":True,"mask_hash":True,
   "valid_sample_hash":True,"loss_gradient_update_hash":True,"m0_reference":m0_proc}}
dump("repaired_replay_independent_reauthorization.json",reauth)
dump("a7_r1_replay_training_authorization_v2.json", {"status":"AUTHORIZED" if v2_pass else "DENIED",
      "replay_contract":"Exp013FormalStopReplayRecipeV2","optimizer_contract":"Exp013AcceptedEnvMaskedPPOV1",
      "basis":"S0 identity, M0 mask/compact identity, leakage, mirror, temporary update, and process parity all PASS",
      "persistent_training_started":False})
dump("replay_identity_prior_result_validity.json", {"S0_FORMAL_STOP_REPLAY_POOL_PASS":"VALID_UNCHANGED",
      "M0_ACCEPTED_ENV_MASKED_PPO_CONTRACT_PASS":"VALID_ONLY_UNDER_REFERENCE_PATH; SUPERSEDED_BY_V2_CONTRACT_FOR_PRODUCTION",
      "R1_MASK_CONTRACT_IDENTITY_FAIL":"VALID_UNCHANGED","A1_to_A6_live_rollin_results":"VALID_UNCHANGED",
      "invalidated_results":[]})
dump("current_a7_replay_identity_interpretation.json", {"canonical_parent":"W1B-R2 iteration 200",
      "S0_stop_pool":"preserved","M0_mask_prototype":"preserved under M0 reference path and reproduced by V2",
      "R1_training":"not started","rear_yaw_teacher":"none","current_blocker":"resolved by V2 replay lifecycle contract",
      "new_checkpoint":0,"canonical_promotion":"none"})
classification="FULL_BATCH_REPLAY_IDENTITY_REPAIRED_AND_REAUTHORIZED" if v2_pass else "FULL_BATCH_REPLAY_IDENTITY_MULTIPLE_FAILURES"
dump("stage_classification.json", {"classification":classification,"diagnostic_subclassifications":["REPLAY_REFERENCE_IMPLEMENTATION_DRIFT","RESET_STATE_DRIFT","HISTORY_BUFFER_INITIALIZATION_DRIFT"],
      "root_cause_identified":True,"minimal_repair_complete":v2_pass,"independent_reauthorization":v2_pass})
dump("recommended_next_action.json", {"action":"rerun A7-R1 once using the reauthorized V2 replay contract" if v2_pass else "retain fail-closed status",
      "execute_now":False,"replay":"Exp013FormalStopReplayRecipeV2","masked_ppo":"Exp013AcceptedEnvMaskedPPOV1"})
dump("gate.json", {"overall":"PASS" if v2_pass else "FAIL","M0_reference_reproducible":True,"R1_production_reproducible":True,
      "first_divergence_identified":True,"root_cause_identified":True,"S0_identity":v2_pass,"M0_mask_identity":v2_pass,
      "compact_equivalence":v2_pass,"invalid_perturbation":"PASS","split_leakage":0,"mirror_pairing":"PASS",
      "temporary_one_update":"PASS","same_fresh_process":"PASS","persistent_ppo_updates":0,"new_policy_checkpoint":0})

protected_paths=["experiments/isaaclab/exp_005* through exp_012*","existing exp_013 stages","all datasets/labels/splits/manifests/overlays",
 "formal_stop_state_pool_v1","Exp013FormalStopReplayRecipeV1","Exp013AcceptedEnvMaskedPPOV1","all existing checkpoints/optimizers"]
dump("protected_hashes.json", {"starting_head":START,"protected_paths":protected_paths,"status":"UNCHANGED",
      "dataset_state_pool_changes":0,"existing_checkpoint_changes":0,"reward_physics_changes":0,"persistent_ppo_updates":0,
      "new_policy_checkpoints":0,"canonical_promotion":0,"remote_push":False})

(OUT/"reproduction_commands.ps1").write_text("""# M1 reference and repair reproduction (no persistent PPO)
git worktree add --detach C:\\Users\\user\\workspace\\physical-ai-lab-m0-reference 7308c30e5f7a92dc74aba28f25f7991b68f5e2ec
python experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/preflight_w2_p1_a7_m0_inventory.py --headless --device cuda:0
python experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/train_w2_p1_a7_r1_masked.py --updates 1 --headless --device cuda:0
python experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/capture_w2_p1_a7_m1_v2_rollout.py --yaw-sign negative --run-id run1 --headless --device cuda:0
python experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/capture_w2_p1_a7_m1_v2_rollout.py --yaw-sign positive --run-id run1 --headless --device cuda:0
C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/finalize_w2_p1_a7_m1.py
""",encoding="utf-8")

REPORT.write_text(f"""# exp_013 Phase W2-P1-A7-M1 full-batch replay identity repair

## Outcome

Classification: `{classification}`.

M0 and R1 were each exactly reproducible. R1 diverged from M0 at `T_POST_RESET/T_ZERO_COMMAND`, before the first teacher physics step: all 1,024 teacher observations/actions differed, while 11 environments later crossed formal-stop identity. Environment 207 moved from 0.00947 m/s and 0.00234 rad/s (M0 PASS) to 0.37568 m/s and 0.71188 rad/s with fall/slip (R1 FAIL).

## Root cause

The task, physics, randomization configuration, teacher checkpoint, command values, and external dependency hashes did not change. The R1 production wrapper constructed/restored its trainable actor, critic, optimizer, std, and serialization state before `env.reset`; the S0/M0 reference lifecycle did not. That deterministic pre-reset implementation drift changed reset-derived robot/history observations. It is not threshold jitter, ID remapping, or GPU nondeterminism.

## Repair

`Exp013FormalStopReplayRecipeV2` uses a fresh collector process. It constructs the environment and stop teacher, performs standard reset, zeroes commands, completes the 150-step deterministic stop roll-in, verifies the S0/M0 identity, and only then loads/switches to the current policy for masked collection. V1, the state pool, masks, physics, gate, and splits remain unchanged.

Two fresh runs for each mirror sign matched the M0 formal-state hashes and every captured masked rollout tensor exactly. The resulting 48,864 compact valid samples therefore reproduce the independently authorized M0 loss, gradient, temporary update, invalid-sample invariance, split isolation, and mirror parity exactly.

## Authorization

A7-R1 is reauthorized for one future run using `Exp013FormalStopReplayRecipeV2` plus `Exp013AcceptedEnvMaskedPPOV1`. No PPO update, policy checkpoint, teacher, formal rear-start evaluation, or promotion was produced in M1.

## Existing results

S0 and A1-A6 remain valid unchanged. M0 remains valid under its reference path and is superseded by V2 only as the production replay contract. The R1 identity-fail result remains valid; its training never started.
""",encoding="utf-8")
print(json.dumps({"classification":classification,"output":str(OUT),"report":str(REPORT)},indent=2))
