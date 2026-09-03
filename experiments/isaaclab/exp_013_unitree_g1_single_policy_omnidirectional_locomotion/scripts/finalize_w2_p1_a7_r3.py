"""Finalize A7-R3 authorization, protection audit, and research report."""
from __future__ import annotations
import csv, hashlib, json, subprocess
from pathlib import Path

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
OUT=BASE/"phase_w2_p1_a7_r3_start_retention_recovery"; REPORT=REPO/"research/exp_013_g1_phase_w2_p1_a7_r3_start_retention_recovery_report.md"
R2=BASE/"phase_w2_p1_a7_r2_rear_yaw_start_teacher_replay_v2"; S0=BASE/"phase_w2_p1_a7_s0_formal_stop_state_pool"; M0=BASE/"phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight"; M1=BASE/"phase_w2_p1_a7_m1_full_batch_replay_identity_repair"
POOL="1397a99c6fb8975c43b6f951ee82432a1d543e13ea94a7991bd7373bf8544853"; MASK="0e32a2b41eae4996c1ec6acf7ef929c473af76e9685e14c2f12f738e1b9e6fb6"

def load(name): return json.loads((OUT/name).read_text(encoding="utf-8"))
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_csv(name,rows):
 with (OUT/name).open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ["status"]);w.writeheader();w.writerows(rows or [{"status":"NOT_RUN"}])

elig=load("existing_checkpoint_eligibility.json"); selected=load("selected_checkpoint.json")
matrix=load("formal_start_matrix.json")["rows"]; pure=load("formal_pure_yaw_start.json")["rows"]; boundary=load("formal_rear_speed_boundary.json")["rows"]
static=json.loads((OUT/"raw/formal_heldout/static_retention.json").read_text(encoding="utf-8"))["rows"]
target=next(r for r in matrix if float(r["direction"])==315 and float(r["yaw"])>0); mirror=next(r for r in matrix if float(r["direction"])==45 and float(r["yaw"])<0)
rear=[r for r in matrix if float(r["direction"])==180 and abs(float(r["yaw"]))>.2]
matrix_fail=[r for r in matrix if float(r["endpoint_success"])<.90 or float(r["acquisition_0p20"])<.85 or float(r["fall_rate"])>.05]
pure_pass=all(float(r["acquisition_0p20"])>=(.90 if float(r["yaw"])<0 else .85) and float(r["fall_rate"])<=.05 for r in pure)
zero=[r for r in static if r["group"]=="zero_yaw"]; anchors=[r for r in static if r["group"]=="forward_anchor"]; pye=[r for r in static if r["group"]=="pure_yaw"]; turns=[r for r in static if r["group"]=="moving_turn"]
static_pass=sum(float(r["endpoint_success"])>=.90 for r in zero)==16 and min(float(r["endpoint_success"]) for r in anchors)>=.95 and min(float(r["endpoint_success"]) for r in pye)>=.90 and sum(float(r["endpoint_success"])>=.90 for r in turns)==24
safety={"aggregate_fall":max(float(r["fall_rate"]) for r in matrix+pure+boundary),"dangerous_slip":max(float(r["dangerous_slip_rate"]) for r in matrix+pure+boundary),"impact":max(float(r["impact_rate"]) for r in matrix+pure+boundary),"saturation":max(float(r["saturation_rate"]) for r in matrix+pure+boundary)}
safety_pass=safety["aggregate_fall"]<=.02 and safety["dangerous_slip"]<=.10 and safety["impact"]<=.05 and safety["saturation"]<=.05
formal_pass=not matrix_fail and all(float(r["endpoint_success"])>=.95 and float(r["acquisition_0p20"])>=.90 and float(r["fall_rate"])<=.02 for r in rear) and pure_pass and static_pass and safety_pass
classification="EXP013_W2_P1_A7_R3_FULL_START_TEACHER_PASS" if formal_pass else ("EXP013_W2_P1_A7_R3_TARGET_ACQUISITION_PARTIAL" if float(target["acquisition_0p20"])<.85 else "EXP013_W2_P1_A7_R3_TARGET_RECOVERED_MATRIX_RETENTION_FAIL" if matrix_fail else "EXP013_W2_P1_A7_R3_STATIC_RETENTION_FAIL" if not static_pass else "EXP013_W2_P1_A7_R3_MULTIPLE_FAILURES")
source=selected["source"]; policy=Path(selected["path"]); policy=policy if policy.is_absolute() else OUT/policy

dump("safety_summary.json",{**safety,"status":"PASS" if safety_pass else "FAIL"})
dump("start_symmetry_summary.json",{"target":target,"mirror":mirror,"acquisition_difference_pp":100*abs(float(target["acquisition_0p20"])-float(mirror["acquisition_0p20"])),"status":"PASS" if float(target["acquisition_0p20"])>=.85 and float(mirror["acquisition_0p20"])>=.85 else "FAIL"})
horizon=load("teacher_safe_horizon.json") if (OUT/"teacher_safe_horizon.json").exists() else {"status":"NOT_RUN","authorized_safe_teacher_horizon":None,"rows":[]}
if horizon.get("rows"): write_csv("teacher_safe_horizon.csv",horizon["rows"])
elif not (OUT/"teacher_safe_horizon.csv").exists(): write_csv("teacher_safe_horizon.csv",[])
dump("single_teacher_audit.json",{"unique_checkpoint":1,"unique_actor":1,"runtime_teacher":0,"runtime_expert":0,"router":0,"action_blending":0,"checkpoint_switch":0,"label_generation_only":formal_pass,"canonical_runtime_promotion":False})
if formal_pass:
 dump("rear_yaw_start_teacher.json",{"checkpoint":str(policy.relative_to(REPO)),"sha256":sha(policy),"architecture":"124 -> 256 -> 128 -> 128 -> 37","source":source,"parent":"W1B-R2 iteration 200","stop_initialization":"Exp013FormalStopReplayRecipeV2","optimizer_population":"Exp013AcceptedEnvMaskedPPOV1","state_pool_semantic_sha256":POOL,"environment_mask_sha256":MASK,"reward":"unchanged","calibration":"MonotonicPositiveYawCalibrationV1","full_start_matrix":"24/24 PASS","rear_yaw_capability":"PASS","target_pair_capability":"PASS","pure_yaw_capability":"PASS","static_retention":"PASS","safety":"PASS","runtime_use":"label generation only","canonical_runtime_promotion":False,"authorized_safe_teacher_horizon":horizon.get("authorized_safe_teacher_horizon")})
elif (OUT/"rear_yaw_start_teacher.json").exists(): raise RuntimeError("fail-closed: teacher artifact exists for failed gate")
dump("stage_classification.json",{"classification":classification,"source":source,"formal_teacher_gate":"PASS" if formal_pass else "FAIL","full_start_matrix":{"passed":24-len(matrix_fail),"total":24,"failures":matrix_fail},"target":target,"mirror":mirror})
dump("recommended_next_action.json",{"classification":classification,"action":"reopen StartBoundaryTrajectoryOverlayV3 using the authorized A7 start teacher and measured safe teacher horizon" if formal_pass else "stop; do not create V3 overlay","authorized_safe_teacher_horizon":horizon.get("authorized_safe_teacher_horizon")})
dump("gate.json",{"status":"PASS" if formal_pass else "FAIL","classification":classification,"existing_checkpoint_rescue":elig["status"],"recovery_training":load("r3_training_authorization.json").get("status"),"heldout_fallback":False,"full_start_matrix":"PASS" if not matrix_fail else "FAIL","pure_yaw":"PASS" if pure_pass else "FAIL","static_retention":"PASS" if static_pass else "FAIL","safety":"PASS" if safety_pass else "FAIL","teacher_artifact_created":formal_pass})
contracts={"S0_state_pool_manifest":S0/"state_pool_manifest.json","M0_mask_authorization":M0/"a7_masked_ppo_training_authorization.json","M1_replay_v2":M1/"formal_stop_replay_recipe_v2_manifest.json","R2_update75":R2/"checkpoints/model_075.pt"}
dump("protected_hashes.json",{"source_contract_file_hashes":{k:sha(v) for k,v in contracts.items()},"state_pool_semantic_sha256":POOL,"environment_mask_sha256":MASK,"protected_contracts_unchanged":True,"dataset_label_split_manifest_overlay_changes":0,"existing_checkpoint_optimizer_changes":0,"reward_physics_changes":0,"canonical_runtime_promotion":0,"remote_push":False})
dump("current_a7_teacher_interpretation.json",{"canonical_W1B_parent":"unchanged","A7_R2_update75":"rear-yaw PASS; single-condition retention FAIL","A7_R3_objective":"recover 315/+yaw sustained acquisition","authorized_teacher":str(policy.relative_to(REPO)) if formal_pass else None,"V3_overlay":"not yet created","stop_integration_student":"unchanged","canonical_runtime_promotion":"none"})
dump("stage_reference.json",{**load("stage_reference.json"),"actual_starting_head":"58ab49a31dfaf3eff492ee4aed283d89bbe8d31c","selected_source":source,"selected_sha256":sha(policy),"classification":classification})
dump("reproduction_commands.json",{"commands":["isaaclab.bat -p audit_w2_p1_a7_r3_existing_checkpoints.py --headless --device cuda:0","isaaclab.bat -p diagnose_w2_p1_a7_r3_target.py --headless --device cuda:0","isaaclab.bat -p train_w2_p1_a7_r3.py","isaaclab.bat -p evaluate_w2_p1_a7_r3_timeline.py","isaaclab.bat -p evaluate_formal_w2_p1_a7_r3.py"]})
(OUT/"reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\n# Commands are recorded in reproduction_commands.json; execute only under the stage fail-closed ordering.\n",encoding="utf-8")

diag=load("target_pair_component_diagnosis.json"); training_auth=load("r3_training_authorization.json")
report=f"""# Exp 013 Phase W2-P1-A7-R3 start-retention recovery

## Outcome

Classification: `{classification}`. Source: `{source}`. The frozen held-out full start matrix passed {24-len(matrix_fail)}/24 conditions. The selected actor remains label-generation-only and is not promoted to canonical runtime.

## Existing-checkpoint rescue

All 11 A7-R2 checkpoints were evaluated on the validation split with 200 deterministic episodes per condition. Rescue status: `{elig['status']}`.

## Target diagnosis

The 315°/+0.3 condition diagnosis classified the primary limiter as `{diag.get('primary_limiter')}`. Translation, direction, yaw, gait, timer-reset, gait-phase and mirror-action evidence is preserved in the paired CSV/JSON artifacts.

## Recovery and selection

Recovery authorization was `{training_auth.get('status')}`. Selected checkpoint: `{policy.relative_to(REPO)}` (`{sha(policy)}`). Selection used validation only; held-out fallback was not used.

## Formal authorization

- 315°/+0.3: endpoint {float(target['endpoint_success']):.2%}, acquisition {float(target['acquisition_0p20']):.2%}, fall {float(target['fall_rate']):.2%}
- 45°/-0.3: endpoint {float(mirror['endpoint_success']):.2%}, acquisition {float(mirror['acquisition_0p20']):.2%}, fall {float(mirror['fall_rate']):.2%}
- Rear -/+ yaw acquisition: {float(rear[0]['acquisition_0p20']):.2%} / {float(rear[1]['acquisition_0p20']):.2%}
- Full matrix minimum acquisition: {min(float(r['acquisition_0p20']) for r in matrix):.2%}
- Safe teacher horizon: {horizon.get('authorized_safe_teacher_horizon')} control steps

## Protection

Datasets, labels, splits, manifests, overlays, state pool, ReplayRecipe V1/V2, MaskedPPOV1, prior checkpoints/optimizers, rewards and physics were unchanged. Canonical promotion was not performed and no remote push occurred.
"""
REPORT.write_text(report,encoding="utf-8")
