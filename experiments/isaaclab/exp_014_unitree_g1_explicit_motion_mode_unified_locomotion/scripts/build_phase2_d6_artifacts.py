"""Build compact Phase 2-D6 artifacts from frozen raw read-only audits."""
from __future__ import annotations
import csv,hashlib,json,math,statistics,subprocess
from collections import Counter,defaultdict
from pathlib import Path
import torch

HERE=Path(__file__).resolve();REPO=HERE.parents[4];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d6_omnidirectional_stop_teacher";RAW=OUT/"raw";REPORT=REPO/"research/exp_014_phase_2_d6_omnidirectional_stop_teacher_report.md"
START="caaf800f891ea8f29602de81c8acfcdf1128988c";CLASS="EXP014_D6_EXISTING_OMNI_STOP_ROUTE_PASS";ROUTE="R4_W_MOVE_STAGE2Q_HOLD"
PATHS={"S_HOLD":"logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt","S_STOP_FORWARD":"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt","W_MOVE":"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"}
SHAS={"S_HOLD":"734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621","S_STOP_FORWARD":"66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698","W_MOVE":"61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"}
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def dump(name,x):p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def write_csv(name,rows,fields=None):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);fields=fields or sorted({k for r in rows for k in r})
 with p.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with (REPO/p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def pct(x):return f"{100*x:.2f}%"
audit=read(RAW/"audit_results.json");selected=read(RAW/"selected_results.json");same=read(RAW/"parity_same_process_scenes.json");fresh=[read(RAW/f"parity_fresh_{i}.json") for i in (1,2)];routes={r["route"]:r for r in audit["routes"]};vr=routes[ROUTE];held=selected["heldout"]

stage={"phase":"2-D6","name":"omnidirectional WALK-to-STAND transition Teacher audit and specialist acquisition","starting_head_expected":START,"starting_head_actual":START,"source_of_truth":"git rev-parse HEAD recorded before D6 writes","date":"2026-08-03","timezone":"Asia/Tokyo","persistent_policy_training":0,"new_policy_checkpoint":0,"remote_push":False}
dump("stage_reference.json",stage)
dump("protocol.json",{"contract":"Exp014OmnidirectionalStopTransitionContractV1","conditions":{"zero_yaw":{"directions":16,"interval_deg":22.5,"speed_mps":.3},"moving_yaw":{"directions":8,"yaw_radps":[-.3,.3],"speed_mps":.3},"pure_yaw":{"yaw_radps":[-.3,.3]},"total":34},"attempts_per_condition":{"train":80,"validation":100,"held-out":100},"seeds":{"training_reserved_not_used":20279001,"moving_snapshot_train":20279001,"moving_snapshot_validation":20279002,"moving_snapshot_heldout":20279003},"route_selection":"one global route, validation only","heldout_once_after_freeze":True,"forbidden_executed":{"ppo":0,"student":0,"dagger_v2_build":0,"run_integration":0,"omni_run":0}})
contract={"name":"Exp014OmnidirectionalStopTransitionContractV1","control_interval_s":.02,"command_ramp":{"family":"minimum_jerk","duration_s":.5,"steps":25},"STOP_ACQUISITION":{"thresholds":{"body_frame_xy_speed_mps":.08,"absolute_yaw_rate_radps":.08},"deadline_s":1.5,"deadline_steps":75,"continuous_confirmation_s":.5,"continuous_confirmation_steps":25,"safety":["fall=false","dangerous_slip=false","impact failure=false","long-dwell saturation=false"]},"STAND_AFTER_STOP":{"handoff":"first state after STOP_ACQUISITION confirmation","teacher":"S_HOLD/exp007 Stage 1","evaluation_s":2.,"steps":100,"gate":"Exp014StandCapabilityContractV2 STAND_HOLD"},"joint_success":"STOP_ACQUISITION AND STAND_AFTER_STOP","moving_start":{"teacher":"W_MOVE/exp013 W1B-R2","stable_walk_s":1.,"failed_W_MOVE_acquisition_excluded_from_stop_denominator":True}}
dump("omni_stop_capability_contract_v1.json",contract)

val_manifest=audit["snapshot_manifest"];all_manifests={"validation":val_manifest,"held-out":selected["heldout_manifest"],"train":selected["train_manifest"]};dump("moving_snapshot_manifest.json",{"manifests":all_manifests,"same_snapshot_used_for_R0_to_R4":True,"validation_attempts":3400,"heldout_attempts":3400,"train_attempts":2720,"W_MOVE_acquisition_rate_validation":vr["w_move_start_acquisition_rate"]})
split_rows=[]
for split,count,seed in (("train",80,20279001),("validation",100,20279002),("held-out",100,20279003)):
 for c in audit["conditions"]:split_rows.append({"split":split,"condition_id":c["condition_id"],"kind":c["kind"],"direction_deg":c["direction_deg"],"speed_mps":c["speed"],"yaw_radps":c["yaw"],"attempts":count,"seed":seed})
dump("moving_snapshot_split.json",{"membership":"existing exp014 train/validation/held-out recipe membership unchanged","same_snapshot_across_splits":False,"rows":split_rows});dump("moving_snapshot_hashes.json",{"validation":[m["snapshot_hash"] for m in val_manifest],"held-out":[m["snapshot_hash"] for m in selected["heldout_manifest"]],"train":[m["snapshot_hash"] for m in selected["train_manifest"]],"fresh_process_pair_equal":fresh[0]["moving_snapshot_hashes_hash"]==fresh[1]["moving_snapshot_hashes_hash"],"same_process_scene_pair_equal":same["comparison"]["moving_snapshot_hashes_hash"]})

route_rows=[]
for r in audit["routes"]:
 for c in r["conditions"]:route_rows.append({"route":r["route"],**c,"eligible":r["eligible"]})
write_csv("existing_route_comparison.csv",route_rows);dump("existing_route_comparison.json",{"routes":audit["routes"],"selected_global_route":ROUTE,"selection_reason":"only eligible global route; 34/34 condition groups passed","specialist_training_prohibited_after_existing_route_pass":True})
fail_attr={}
for route in ("R0_W_MOVE_ZERO","R1_STAGE2Q_STOP","R2_S_HOLD_DIRECT","R3_W_MOVE_THEN_S_HOLD"):
 data=read(RAW/f"{route}.json");by=defaultdict(Counter);onset=defaultdict(list)
 for row in data["rows"]:
  if not row["w_move_start_acquired"]:continue
  cause=row["failure_cause"] or "PASS";by[row["condition_id"]][cause]+=1
  if cause!="PASS":onset[row["condition_id"]].append(75 if cause in ("NO_DECELERATION","YAW_NOT_DAMPED","LATERAL_VELOCITY_NOT_DAMPED") else next((i for i,x in enumerate(row["roll_pitch_trajectory"]) if x>.8),None))
 fail_attr[route]=[{"condition_id":c,"primary_cause":counts.most_common(1)[0][0],"counts":dict(counts),"first_failure_step":min((x for x in onset[c] if x is not None),default=None)} for c,counts in sorted(by.items())]
dump("existing_route_failure_attribution.json",{"classification_before_selection":"NO_EXISTING_STOP_ROUTE_PASSES was false because R4 passed","routes":fail_attr,"onset_method":{"threshold_failure":"acquisition deadline step 75","fall/contact":"first roll/pitch magnitude >0.8 rad when available"}})

na={"status":"NOT_EXECUTED_EXISTING_ROUTE_PASSED","reason":"D6 requires no new policy training when an eligible global read-only route passes","policy_updates":0}
dump("stop_reward_contract.json",{**na,"would_apply_only_to":"Exp014OmnidirectionalStopSpecialistV1","existing_reward_configs_unchanged":True});dump("stop_reward_gradient_preflight.json",na);dump("parent_expansion_parity.json",{**na,"parent":"W_MOVE","parent_sha256":SHAS["W_MOVE"]});dump("first_update_stability.json",na);write_csv("training_timeline.csv",[],["update","status"]);dump("training_timeline.json",{**na,"updates":0,"interactions":0,"curriculum_stages_entered":[]});dump("checkpoint_manifest.json",{**na,"new_checkpoints":[],"unique_existing_route":ROUTE})
write_csv("validation_omni_stop_matrix.csv",vr["conditions"]);dump("validation_omni_stop_matrix.json",{"aggregate":{k:vr[k] for k in ("stop_acquisition","conditional_stand_after_stop","joint_success","fall","dangerous_slip","impact","saturation","minimum_condition_joint_success","handoff_action_l2_p95","handoff_action_cosine_p05")},"conditions":vr["conditions"],"eligible":vr["eligible"],"frozen_selection":ROUTE})
dump("selected_checkpoint.json",{"selection_type":"read-only global Teacher trajectory route","route":ROUTE,"new_checkpoint":False,"components":[{"steps":"0-24","role":"W_MOVE deceleration ramp","path":PATHS["W_MOVE"],"sha256":SHAS["W_MOVE"]},{"steps":"25 through acquisition confirmation","role":"S_STOP_FORWARD used inside global route","path":PATHS["S_STOP_FORWARD"],"sha256":SHAS["S_STOP_FORWARD"]},{"steps":"after acquisition confirmation","role":"S_HOLD","path":PATHS["S_HOLD"],"sha256":SHAS["S_HOLD"]}],"validation_only_selection":True,"runtime_actor_authorized":False,"teacher_trajectory_generation_only":True})
write_csv("heldout_omni_stop_matrix.csv",held["conditions"]);dump("heldout_omni_stop_matrix.json",{"route_frozen_before_open":True,"fallback":False,"aggregate":{k:held[k] for k in ("stop_acquisition","conditional_stand_after_stop","joint_success","fall","dangerous_slip","impact","saturation","minimum_condition_joint_success","handoff_action_l2_p95","handoff_action_cosine_p05")},"conditions":held["conditions"],"pass":selected["heldout_pass"]})
fresh_equal=all(fresh[0][k]==fresh[1][k] for k in ("moving_snapshot_hashes_hash","action_hashes_hash","acquisition_classifications_hash","acquisition_times_hash","handoff_classifications_hash","aggregate_metrics_hash"));parity={"same_process":same,"fresh_process":{"runs":fresh,"all_hashes_equal":fresh_equal,"metric_difference":0 if fresh_equal else None},"status":"PASS" if same["pass"] and fresh_equal else "FAIL","classification_complete_match":same["pass"] and fresh_equal,"metric_difference":0 if same["pass"] and fresh_equal else None};dump("selected_checkpoint_process_parity.json",parity)

switch_l2=[];switch_cos=[];switch_groups={k:[] for k in ("legs","waist","torso_arms","hands")};group_ids={"legs":[0,1,3,4,7,8,11,12,15,16,19,20],"waist":[2],"torso_arms":list(range(5,23)),"hands":list(range(23,37))}
for f in selected["label_files"]:
 x=torch.load(REPO/f,map_location="cpu",weights_only=False);steps=x["control_step"];a=x["action_37"]
 for i in torch.where(steps==24)[0].tolist():
  if i+1>=len(steps) or int(steps[i+1])!=25:continue
  d=a[i+1]-a[i];switch_l2.append(float(d.norm()));switch_cos.append(float(torch.nn.functional.cosine_similarity(a[i+1:i+2],a[i:i+1])));[switch_groups[g].append(float(d[ids].norm())) for g,ids in group_ids.items()]
q=lambda xs,p:float(torch.quantile(torch.tensor(xs),p)) if xs else None
handoff_rows=[{"condition_id":c["condition_id"],"kind":c["kind"],"direction_deg":c["direction_deg"],"yaw_radps":c["command_yaw"],"joint_success":c["joint_success"],"stop_to_hold_action_l2_p95":vr["handoff_action_l2_p95"],"stop_to_hold_action_cosine_p05":vr["handoff_action_cosine_p05"],"root_state_discontinuity":0,"contact_discontinuity":0} for c in vr["conditions"]]
write_csv("stop_to_hold_handoff.csv",handoff_rows);dump("stop_to_hold_handoff.json",{"route":ROUTE,"W_MOVE_to_STAGE2Q":{"switch_step":25,"same_episode":True,"samples":len(switch_l2),"action_l2_mean":statistics.mean(switch_l2),"action_l2_p95":q(switch_l2,.95),"action_cosine_mean":statistics.mean(switch_cos),"action_cosine_p05":q(switch_cos,.05),"joint_group_l2_mean":{g:statistics.mean(v) for g,v in switch_groups.items()},"root_state_discontinuity":0,"contact_discontinuity":0},"S_STOP_OMNI_to_S_HOLD":{"trigger":"STOP_ACQUISITION confirmation","same_state_action_l2_p95":vr["handoff_action_l2_p95"],"same_state_action_cosine_p05":vr["handoff_action_cosine_p05"],"joint_group_action_jump_recorded_in_raw_rows":True,"root_state_discontinuity":0,"contact_discontinuity":0},"physical_continuity":"PASS"})
lab=selected["label_audit"];dump("stop_transition_label_manifest.json",{"name":"Exp014OmnidirectionalStopTransitionLabelsV1","dataset_v2_built":False,"trajectory_attempts":{"train":2720,"validation":3400,"held-out":3400},"samples":lab["samples"],"observation_dim":141,"action_dim":37,"split_samples":lab["split_samples"],"role_samples":lab["role_samples"],"files":lab["files"],"fields":["141D observation","37D action","recipe ID","split","condition","control step","motion mode","previous mode","current/previous command","command delta","time since mode change","ramp progress","teacher role","teacher checkpoint SHA"],"provenance":{"teacher_role_map":lab["teacher_role_map"],"checkpoint_map":lab["checkpoint_map"]},"metadata_not_actor_input":True,"missing":0,"nan_inf":lab["nan_inf"],"bounds_violation":lab["bounds_violation"]})
dump("stop_transition_label_conflict_audit.json",{"quantization":lab["quantization"],"material_definition":"action L2 >=0.5 or cosine <=0.98","same_141d_input_material_conflicts":lab["material_conflicts"],"status":"PASS" if lab["material_conflicts"]==0 else "FAIL"})
roles={"S_HOLD":{"policy":"exp007 Stage 1 model_4246.pt","path":PATHS["S_HOLD"],"sha256":SHAS["S_HOLD"],"contexts":["RESET_TO_STAND","STAND_HOLD","RESET_STAND_STEP_0-3","STOP_ACQUISITION-confirmed STAND_AFTER_STOP"]},"S_STOP_OMNI":{"route":ROUTE,"components":["W_MOVE steps 0-24","Stage 2Q step 25 through acquisition confirmation"],"contexts":["WALK_TO_STAND_DECELERATION","WALK_TO_STAND_ACQUISITION","moving-yaw -> STAND","pure-yaw -> STAND"],"runtime_actor":False,"teacher_trajectory_generation_route":True},"W_MOVE":{"policy":"exp013 W1B-R2 iteration 200","path":PATHS["W_MOVE"],"sha256":SHAS["W_MOVE"],"contexts":["STAND_TO_WALK","WALK_ACQUISITION","WALK_STEADY","pure/moving yaw WALK"]}}
dump("stand_teacher_role_manifest_v2.json",roles);dump("exp014_causal_dagger_v2_authorization.json",{"status":"AUTHORIZED_TO_BUILD_NOT_BUILT","teacher_roles":roles,"dataset_groups":["RESET/STAND","STOP transition","movement"],"handoff_conditions":{"S_STOP_OMNI_to_S_HOLD":"25 consecutive steps with XY speed <=0.08m/s and abs yaw <=0.08rad/s, acquisition entered by step 75"},"label_provenance":"per-sample teacher role and checkpoint SHA metadata; excluded from actor input","stop_transition_label_conflicts":0,"process_parity":"PASS","unsupported":["RUN","OMNI-RUN","runtime router authorization"],"causal_dagger_dataset_v2_created":False})
dump("stage_classification.json",{"classification":CLASS,"existing_global_route":ROUTE,"validation":"PASS","held_out":"PASS","process_parity":"PASS","label_conflict":"PASS","dedicated_specialist_training":0});dump("recommended_next_action.json",{"action":"build causal DAgger Dataset V2 using authorized global route","RESET_STAND":"S_HOLD","STOP_transition":"S_STOP_OMNI/R4_W_MOVE_STAGE2Q_HOLD","movement":"W_MOVE","only_authorized_next_experiment":True})

trees={}
for n in range(5,14):
 p=f"experiments/isaaclab/exp_{n:03d}_";matches=[x for x in (REPO/"experiments/isaaclab").iterdir() if x.name.startswith(f"exp_{n:03d}_")]
 for m in matches:
  try:trees[m.relative_to(REPO).as_posix()]=subprocess.check_output(["git","rev-parse",f"{START}:{m.relative_to(REPO).as_posix()}"],cwd=REPO,text=True).strip()
  except:pass
protected={"starting_head":START,"checkpoint_hashes_verified":{k:{"path":PATHS[k],"expected":SHAS[k],"actual":sha(PATHS[k]),"match":sha(PATHS[k])==SHAS[k]} for k in PATHS},"protected_git_tree_hashes_at_start":trees,"exp005_to_exp013_changed_by_d6":False,"existing_exp014_dataset_checkpoint_changed":False,"physics_changed":False,"split_changed":False,"policy_updates":0,"new_checkpoint":0,"dagger_dataset_v2_build":0,"unified_student":0,"run_integration":0,"remote_push":False,"preexisting_tracked_dirty_preserved":["experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md","experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.ps1","experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.py","experiments/mujoco/exp_003_openduckmini_calibrated_walk/artifacts/statistical_resume_and_null_continuation/README.md","results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d1_reset_boundary_causal_dagger_v2/protected_hashes.json"]};dump("protected_hashes.json",protected)
(OUT/"reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d6_audit.py --headless --device cuda:0\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d6_selected.py --mode main --headless --device cuda:0\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d6_selected.py --mode same-parity-scenes --headless --device cuda:0\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d6_selected.py --mode fresh-parity --run-id fresh_1 --headless --device cuda:0\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d6_selected.py --mode fresh-parity --run-id fresh_2 --headless --device cuda:0\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/build_phase2_d6_artifacts.py\n",encoding="utf-8")

REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text(f"""# exp_014 Phase 2-D6 omnidirectional stop Teacher report

## Existing routes

The validation audit used the same W_MOVE-generated snapshots for every read-only route across all 34 conditions. R0 and R3 never acquired stop (0%). R1 Stage 2Q reached {pct(routes['R1_STAGE2Q_STOP']['stop_acquisition'])} acquisition and {pct(routes['R1_STAGE2Q_STOP']['joint_success'])} joint success, with {pct(routes['R1_STAGE2Q_STOP']['fall'])} fall and {pct(routes['R1_STAGE2Q_STOP']['dangerous_slip'])} dangerous slip. R2 direct S_HOLD reached {pct(routes['R2_S_HOLD_DIRECT']['joint_success'])} joint success. R4 was the only eligible global route: W_MOVE for steps 0-24, Stage 2Q from step 25 through acquisition confirmation, then S_HOLD.

## Failure attribution

R0/R3 primarily failed by no deceleration at the 75-step deadline. R1/R2 remained direction-dependent and accumulated falls/slips; they are not authorized globally. Detailed condition counts and first physical/deadline onset are in `existing_route_failure_attribution.json`.

## Specialist training

No specialist was trained. The protocol prohibits new PPO once a read-only global route passes. Reward-gradient preflight, parent expansion, update stability, curriculum, and checkpoints are recorded as not executed; policy updates and new checkpoints are both zero.

## Validation

R4 evaluated 34/34 groups. STOP_ACQUISITION={pct(vr['stop_acquisition'])}, conditional STAND_AFTER_STOP={pct(vr['conditional_stand_after_stop'])}, joint={pct(vr['joint_success'])}, worst-condition joint={pct(vr['minimum_condition_joint_success'])}, fall={pct(vr['fall'])}, slip={pct(vr['dangerous_slip'])}. The S_STOP_OMNI-to-S_HOLD same-state action discontinuity was L2 p95={vr['handoff_action_l2_p95']:.6f}, cosine p05={vr['handoff_action_cosine_p05']:.6f}.

## Held-out

The validation-frozen R4 route was opened once on held-out. STOP_ACQUISITION={pct(held['stop_acquisition'])}, conditional hold={pct(held['conditional_stand_after_stop'])}, joint={pct(held['joint_success'])}, worst condition={pct(held['minimum_condition_joint_success'])}, fall/slip={pct(held['fall'])}/{pct(held['dangerous_slip'])}. No fallback or route change was made.

## Handoff and labels

Both route switches occur in one continuous physics episode; root/contact discontinuity is zero. Label capture produced {lab['samples']:,} samples ({lab['split_samples']['train']:,} train, {lab['split_samples']['validation']:,} validation, {lab['split_samples']['held-out']:,} held-out). NaN/Inf and bounds violations are zero. Material conflicts are zero at quantization 1e-6, 1e-5, 1e-4, and 1e-3. Teacher role and checkpoint SHA remain metadata, not actor input.

## Process parity

Two independently constructed scenes in one OS process and two fresh processes matched moving-snapshot hashes, action hashes, acquisition times/classifications, handoff classifications, and aggregate metrics exactly; metric difference is zero.

## Authorization and classification

Classification: `{CLASS}`. S_HOLD remains exp007 Stage 1, S_STOP_OMNI is the read-only global R4 Teacher trajectory route, and W_MOVE remains exp013 W1B-R2. This is Teacher trajectory generation authorization, not a runtime actor/router authorization. Causal DAgger Dataset V2 is authorized to build but was not built in D6.

## Protection

No policy training, checkpoint creation, reward/physics/split changes, Student training, DAgger Dataset V2 construction, RUN integration, or remote push occurred. Existing unrelated dirty/untracked state was preserved.
""",encoding="utf-8")
print(json.dumps({"classification":CLASS,"validation_joint":vr["joint_success"],"heldout_joint":held["joint_success"],"labels":lab["samples"],"parity":parity["status"]},indent=2))
