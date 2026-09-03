"""Finalize D6 after read-only continuity failure and C1 specialist gate failure."""
from __future__ import annotations
import csv,json
from pathlib import Path
HERE=Path(__file__).resolve();REPO=HERE.parents[4];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d6_omnidirectional_stop_teacher";RAW=OUT/"raw";REPORT=REPO/"research/exp_014_phase_2_d6_omnidirectional_stop_teacher_report.md";CLASS="EXP014_D6_STOP_SPECIALIST_VALIDATION_FAIL"
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def csvout(n,rows,fields=None):
 fields=fields or sorted({k for r in rows for k in r})
 with (OUT/n).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
audit=read(RAW/"audit_results.json");train=read(RAW/"training_results.json");first=read(RAW/"training_first_update.json");parent=read(RAW/"training_parent_parity.json");reward=read(RAW/"training_reward_contract.json");grad=read(RAW/"training_reward_gradient_preflight.json");handoff=read(OUT/"stop_to_hold_handoff.json");routes=[]
stage_ref=read(OUT/"stage_reference.json");stage_ref.update({"persistent_policy_training":40,"training_interactions":1904000,"new_policy_checkpoint":5,"selected_policy_checkpoint":0});dump("stage_reference.json",stage_ref)
protocol=read(OUT/"protocol.json");protocol.update({"read_only_routes_complete":True,"existing_route_result":"NO_EXISTING_STOP_ROUTE_PASSES","specialist_training":{"name":"Exp014OmnidirectionalStopSpecialistV1","updates_completed":40,"maximum":200,"stopped_at":"C1 progression gate","rollout_steps":100,"episode_steps":150,"seed":20279001,"learning_rate":1.5e-5,"adaptive_lr":False,"gradient_clipping":10},"heldout_specialist_evaluation":0,"dagger_v2_build":0});protocol["forbidden_executed"]={"student":0,"dagger_v2_build":0,"run_integration":0,"omni_run":0};dump("protocol.json",protocol)
for r in audit["routes"]:
 x={k:v for k,v in r.items() if k!="conditions"};x["physical_gate_before_internal_handoff_audit"]=r["eligible"]
 if r["route"]=="R4_W_MOVE_STAGE2Q_HOLD":x.update({"internal_handoff_action_l2_p95":handoff["W_MOVE_to_STAGE2Q"]["action_l2_p95"],"internal_handoff_action_cosine_p05":handoff["W_MOVE_to_STAGE2Q"]["action_cosine_p05"],"internal_handoff_gate":False,"eligible":False,"failure":"ACTION_DISCONTINUITY at step 25"})
 else:x["eligible"]=False
 routes.append(x)
rows=[]
for raw,x in zip(audit["routes"],routes):
 for c in raw["conditions"]:rows.append({"route":raw["route"],**c,"eligible":x["eligible"],"route_failure":x.get("failure")})
csvout("existing_route_comparison.csv",rows);dump("existing_route_comparison.json",{"routes":routes,"eligible_global_routes":[],"intermediate_classification":"EXP014_D6_NO_EXISTING_STOP_ROUTE_PASSES","specialist_training_started_after_failure_attribution":True,"R4_note":"physical success was 100%, but the mandatory W_MOVE-to-Stage2Q handoff continuity gate failed"})
failure=read(OUT/"existing_route_failure_attribution.json");failure["routes"]["R4_W_MOVE_STAGE2Q_HOLD"]=[{"condition_id":c,"primary_cause":"ACTION_DISCONTINUITY","first_failure_step":25,"action_l2_p95":handoff["W_MOVE_to_STAGE2Q"]["action_l2_p95"],"action_cosine_p05":handoff["W_MOVE_to_STAGE2Q"]["action_cosine_p05"]} for c in range(34)];failure["intermediate_classification"]="EXP014_D6_NO_EXISTING_STOP_ROUTE_PASSES";dump("existing_route_failure_attribution.json",failure)
dump("stop_reward_contract.json",reward);dump("stop_reward_gradient_preflight.json",grad);dump("parent_expansion_parity.json",parent);dump("first_update_stability.json",first)
csvout("training_timeline.csv",train["timeline"]);dump("training_timeline.json",{"status":train["status"],"stop_reason":train["stop_reason"],"updates_completed":40,"interactions":1904000,"curriculum":{"C1_CARDINAL_ZERO_YAW":{"planned":"1-40","executed":"1-40","progression_gate":"FAIL"},"C2_16_DIRECTION_ZERO_YAW":{"executed":0},"C3_MOVING_YAW":{"executed":0},"C4_FULL_BALANCED":{"executed":0}},"timeline":train["timeline"]});dump("checkpoint_manifest.json",{"specialist":"Exp014OmnidirectionalStopSpecialistV1","checkpoints":train["checkpoint_manifest"],"selected":None,"existing_checkpoints_overwritten":False})
vrows=[]
for e in train["evaluations"]:
 for c in e["summary"]["conditions"]:vrows.append({"update":e["update"],**c})
csvout("validation_omni_stop_matrix.csv",vrows);dump("validation_omni_stop_matrix.json",{"evaluations":train["evaluations"],"eligible_updates":[],"heldout_opened_for_specialist":False,"C1_progression_gate":train["evaluations"][-1]["progression_gate"]})
dump("selected_checkpoint.json",{"selected_checkpoint":None,"eligible_checkpoints":0,"reason":"C1 STOP_ACQUISITION remained 0% at updates 0,1,10,20,40","latest_diagnostic_checkpoint":train["checkpoint_manifest"][-1],"latest_checkpoint_not_authorized":True,"heldout_not_opened":True})
csvout("heldout_omni_stop_matrix.csv",[],["status"]);dump("heldout_omni_stop_matrix.json",{"status":"NOT_RUN_NO_ELIGIBLE_SPECIALIST_CHECKPOINT","fallback":False,"checkpoint_or_reward_changed_after_heldout":False,"invalidated_pre-specialist_R4_read_only_probe":"not used for specialist selection or authorization"});dump("selected_checkpoint_process_parity.json",{"status":"NOT_EXECUTED_NO_SELECTED_CHECKPOINT","same_process_runs":0,"fresh_process_runs":0,"authorization":False})
dump("stop_to_hold_handoff.json",{**handoff,"eligibility_interpretation":{"R4":"FAIL due W_MOVE-to-Stage2Q action continuity","specialist":"NOT_EVALUABLE because STOP_ACQUISITION=0%"},"authorized":False})
csvout("stop_to_hold_handoff.csv",[{"transition":"W_MOVE_to_STAGE2Q","step":25,"action_l2_p95":handoff["W_MOVE_to_STAGE2Q"]["action_l2_p95"],"action_cosine_p05":handoff["W_MOVE_to_STAGE2Q"]["action_cosine_p05"],"gate":"FAIL","physical_root_discontinuity":0,"contact_discontinuity":0},{"transition":"S_STOP_OMNI_to_S_HOLD","step":"acquisition confirmation","action_l2_p95":None,"action_cosine_p05":None,"gate":"NOT_EVALUABLE","physical_root_discontinuity":None,"contact_discontinuity":None}])
oldlab=read(OUT/"stop_transition_label_manifest.json");dump("stop_transition_label_manifest.json",{"status":"INVALIDATED_NOT_AUTHORIZED","reason":"labels were captured from R4 before the internal handoff continuity gate was computed; R4 is ineligible","raw_samples_retained_for_audit_only":oldlab.get("samples",oldlab.get("raw_samples_retained_for_audit_only",0)),"dataset_v2_built":False,"authorized_samples":0,"actor_input_contamination":0});oldconf=read(OUT/"stop_transition_label_conflict_audit.json");oldconf.update({"status":"DIAGNOSTIC_ONLY_ROUTE_INELIGIBLE","authorization":False});dump("stop_transition_label_conflict_audit.json",oldconf)
roles={"S_HOLD":{"status":"AUTHORIZED_UNCHANGED","checkpoint":"logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt","sha256":"734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"},"S_STOP_OMNI":{"status":"NOT_AUTHORIZED","existing_routes":"FAIL","dedicated_specialist":"VALIDATION_FAIL"},"W_MOVE":{"status":"AUTHORIZED_UNCHANGED","checkpoint":"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt","sha256":"61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"}}
dump("stand_teacher_role_manifest_v2.json",roles);dump("exp014_causal_dagger_v2_authorization.json",{"status":"DENIED","reason":"S_STOP_OMNI is not authorized","S_HOLD":"AUTHORIZED","S_STOP_OMNI":"DENIED","W_MOVE":"AUTHORIZED","stop_transition_label_conflicts":"not applicable; source route ineligible","process_parity":"not run without selected specialist","causal_dagger_dataset_v2_created":False})
dump("stage_classification.json",{"classification":CLASS,"intermediate_classification":"EXP014_D6_NO_EXISTING_STOP_ROUTE_PASSES","existing_route_gate":"FAIL","specialist_C1_progression_gate":"FAIL","validation":"FAIL","held_out":"NOT_RUN","process_parity":"NOT_RUN","label_authorization":"DENIED"});dump("recommended_next_action.json",{"action":"diagnose worst direction/yaw neighborhood only","do_not_build_dagger_v2":True,"do_not_fallback":True,"only_authorized_next_experiment":True})
prot=read(OUT/"protected_hashes.json");prot.update({"policy_updates":40,"new_checkpoint":{"scope":"D6 S_STOP_OMNI diagnostic only","count":5,"selected":0,"updates":[0,1,10,20,40]},"reward_change_scope":"D6-only Exp014OmniStopRewardV1 config; protected reward configs unchanged","dagger_dataset_v2_build":0,"unified_student":0,"run_integration":0,"remote_push":False});dump("protected_hashes.json",prot)
REPORT.write_text(f"""# exp_014 Phase 2-D6 omnidirectional stop Teacher report

## Existing routes

All 34 conditions were audited from common W_MOVE snapshots. R0 and R3 had 0% STOP_ACQUISITION. R1 Stage 2Q reached 70.83% acquisition and 69.73% joint success, with 30.09% fall and 27.98% dangerous slip. R2 direct S_HOLD reached 69.53% joint success. R4 was physically successful (100% acquisition/hold/joint, 0% fall/slip), but failed the mandatory internal handoff gate: W_MOVE→Stage 2Q at step 25 had action L2 p95 2.3052 (>0.5) and cosine p05 0.9644 (<0.98). Therefore no read-only global route passed.

## Failure attribution

R0/R3 failed by no deceleration at the 75-step acquisition deadline. R1/R2 remained direction-dependent with fall/slip failures. R4's first failure is `ACTION_DISCONTINUITY` at step 25; its root/contact state remained continuous because the switch was in the same episode.

## Specialist training

`Exp014OmnidirectionalStopSpecialistV1` was initialized from W_MOVE. The 124D→141D expansion had max output difference 0, old columns/hidden/output copied, and 17 new columns zero. `Exp014OmniStopRewardV1` reused existing reward families; XY/yaw weights were 8/4. Settling-gradient/total was 1.025 and regularization cosine was -0.998. The one-update gate passed (KL 0.00646, all-step KL 0.04624, clip 0.0761, mean-action shift 0.02258, NaN/Inf 0).

C1 ran 40 updates, 1,904,000 interactions, 100-step rollouts, 150-step episodes, LR 1.5e-5 fixed, and gradient clipping 10. STOP_ACQUISITION stayed 0% at updates 0/1/10/20/40. The C1 progression gate failed, so C2-C4 were not entered. Five D6-only diagnostic checkpoints were retained in raw results; none is selected or authorized.

## Validation

The 16-direction, moving-yaw, and pure-yaw matrix was evaluated at each required checkpoint reached. At update 40 aggregate and minimum-condition acquisition/joint success were 0%; fall/slip were 0%/0%. No checkpoint met the validation gate.

## Held-out and parity

The specialist held-out set was not opened, no fallback was used, and process parity was not run because no checkpoint was selected. An earlier R4 read-only probe is explicitly invalidated and is not used for specialist selection or authorization.

## Labels and authorization

R4 labels captured before its internal handoff gate was calculated are retained as raw diagnostic data only and invalidated for authorization. `S_HOLD` and `W_MOVE` remain unchanged and authorized in their prior scopes. `S_STOP_OMNI` is not authorized, so Causal DAgger Dataset V2 remains denied and was not built.

## Classification

`{CLASS}`

The only recommended next experiment is a worst direction/yaw-neighborhood diagnosis. Do not build DAgger Dataset V2.

## Protection

exp_005-exp_013, existing exp_014 datasets/checkpoints, S_HOLD, W_MOVE, S_STOP_FORWARD, physics, and splits were not modified. Student/DAgger/RUN integration counts are zero. No remote push occurred.
""",encoding="utf-8")
print(json.dumps({"classification":CLASS,"updates":40,"interactions":1904000,"selected":None},indent=2))
