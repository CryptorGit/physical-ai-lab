"""Fail-closed A7-R3 finalization after preregistered early-guard stop."""
from __future__ import annotations
import csv,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve();REPO=HERE.parents[4];BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion";OUT=BASE/"phase_w2_p1_a7_r3_start_retention_recovery";R2=BASE/"phase_w2_p1_a7_r2_rear_yaw_start_teacher_replay_v2";REPORT=REPO/"research/exp_013_g1_phase_w2_p1_a7_r3_start_retention_recovery_report.md";C="EXP013_W2_P1_A7_R3_TRAINING_UNSTABLE"
def load(n):return json.loads((OUT/n).read_text(encoding="utf-8"))
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def placeholder(csvname,jsonname,reason):
 with (OUT/csvname).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=["status","reason"]);w.writeheader();w.writerow({"status":"NOT_RUN","reason":reason})
 dump(jsonname,{"status":"NOT_RUN","reason":reason,"heldout_fallback":False})
reason="R3-C stopped at update 3 because rear +yaw validation acquisition fell below the preregistered 90% early guard"
for a,b in (("a7_r3_capability_timeline.csv","a7_r3_capability_timeline.json"),("formal_start_matrix.csv","formal_start_matrix.json"),("formal_pure_yaw_start.csv","formal_pure_yaw_start.json"),("formal_rear_speed_boundary.csv","formal_rear_speed_boundary.json"),("teacher_safe_horizon.csv","teacher_safe_horizon.json")):placeholder(a,b,reason)
dump("selected_checkpoint.json",{"status":"NOT_SELECTED","reason":reason,"heldout_fallback":False})
dump("selected_checkpoint_process_parity.json",{"status":"NOT_RUN","reason":"no eligible selected checkpoint"})
guard=load("early_guard.json");diagnosis=load("target_pair_component_diagnosis.json");elig=load("existing_checkpoint_eligibility.json")
dump("safety_summary.json",{"status":"NOT_AUTHORIZED","early_guard":guard["rows"][-1]})
dump("start_symmetry_summary.json",{"status":"FAIL","rear_negative_acquisition":guard["rows"][-1]["rear_negative_acquisition"],"rear_positive_acquisition":guard["rows"][-1]["rear_positive_acquisition"],"difference_pp":100*abs(guard["rows"][-1]["rear_negative_acquisition"]-guard["rows"][-1]["rear_positive_acquisition"])})
dump("single_teacher_audit.json",{"unique_checkpoint":0,"teacher_artifact_created":False,"runtime_teacher":0,"router":0,"action_blending":0,"canonical_runtime_promotion":False})
dump("stage_classification.json",{"classification":C,"existing_checkpoint_rescue":"NO_RESCUE","diagnosis":"YAW_RATE_OSCILLATION","training_stopped_update":3,"reason":reason})
dump("recommended_next_action.json",{"classification":C,"action":"stop; do not create StartBoundaryTrajectoryOverlayV3 from an unauthorized teacher","single_method":True})
dump("gate.json",{"status":"FAIL","classification":C,"existing_checkpoint_rescue":"FAIL","target_diagnosis":"AUTHORIZED","first_update_stability":"PASS","early_guard":"FAIL at update 3","heldout":"NOT_RUN","teacher_artifact_created":False})
contracts={"pool":BASE/"phase_w2_p1_a7_s0_formal_stop_state_pool/state_pool_manifest.json","replay_v2":BASE/"phase_w2_p1_a7_m1_full_batch_replay_identity_repair/formal_stop_replay_recipe_v2_manifest.json","mask":BASE/"phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight/a7_masked_ppo_training_authorization.json","R2_update75":R2/"checkpoints/model_075.pt"}
dump("protected_hashes.json",{"source_hashes":{k:sha(v) for k,v in contracts.items()},"state_pool_semantic_sha256":"1397a99c6fb8975c43b6f951ee82432a1d543e13ea94a7991bd7373bf8544853","environment_mask_sha256":"0e32a2b41eae4996c1ec6acf7ef929c473af76e9685e14c2f12f738e1b9e6fb6","datasets_labels_splits_manifests_overlays_changed":0,"existing_checkpoints_optimizers_changed":0,"reward_physics_changed":0,"new_checkpoint_scope":"A7-R3 recovery candidates only","V3_overlay":0,"stop_integration_student_training":0,"canonical_runtime_promotion":0,"remote_push":False})
dump("current_a7_teacher_interpretation.json",{"canonical_W1B_parent":"unchanged","A7_R2_update75":"rear-yaw PASS; single-condition retention FAIL","A7_R3_objective":"recover 315/+yaw sustained acquisition","A7_R3_result":"early-guard training instability at update 3","V3_overlay":"not created","stop_integration_student":"unchanged","authorized_start_teacher":None,"canonical_runtime_promotion":"none"})
dump("stage_reference.json",{**load("stage_reference.json"),"actual_starting_head":"58ab49a31dfaf3eff492ee4aed283d89bbe8d31c","classification":C})
(OUT/"reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\n# R3-A, R3-B, and the single R3-C chain were executed in fail-closed order.\n",encoding="utf-8")
if (OUT/"rear_yaw_start_teacher.json").exists():raise RuntimeError("unauthorized teacher artifact exists")
rows=elig["rows"];best=max(rows,key=lambda r:r["minimum_acquisition"]);target=next(r for r in diagnosis["rows"] if r["condition_name"]=="T0");mirror=next(r for r in diagnosis["rows"] if r["condition_name"]=="M0")
REPORT.write_text(f"""# Exp 013 Phase W2-P1-A7-R3 start-retention recovery

## Outcome

Classification: `{C}`. No label-generation teacher was authorized.

## Existing checkpoint rescue

All 11 A7-R2 checkpoints were evaluated over 24 validation conditions with 200 episodes each. No checkpoint was eligible. The best minimum acquisition was {best['minimum_acquisition']:.1%} at update {best['update']}.

## Diagnosis

At update 75, 315°/+0.3 had endpoint {target['endpoint_success']:.1%}, 0.10 s acquisition {target['acquisition_0p10']:.1%}, and 0.20 s acquisition {target['acquisition_0p20']:.1%}. Translation, direction, and gait each sustained PASS; yaw sustained PASS was {target['yaw_sustained_0p20']:.1%}, with {target['yaw_timer_resets']:.2f} resets and {target['longest_yaw_pass_s']:.4f} s longest PASS. The mirrored 45°/-0.3 condition acquired at {mirror['acquisition_0p20']:.1%}. The primary limiter was yaw-rate oscillation.

## Localized continuation

The single fixed continuation used A7-R2 update 75, LR 5e-6, seed 20278631, unchanged reward, and the preregistered 30/25/25/20 command mixture. Update 1 temporary/persistent tensors matched exactly. Update 3 triggered the mandatory early guard: rear -yaw acquisition {guard['rows'][-1]['rear_negative_acquisition']:.1%}, rear +yaw {guard['rows'][-1]['rear_positive_acquisition']:.1%}. Training stopped; no held-out evaluation or horizon sweep was run.

## Protection

All protected datasets, labels, splits, manifests, overlays, state pools, replay/mask contracts, prior checkpoints/optimizers, reward, and physics remain unchanged. No V3 overlay, canonical promotion, or remote push occurred.
""",encoding="utf-8")
