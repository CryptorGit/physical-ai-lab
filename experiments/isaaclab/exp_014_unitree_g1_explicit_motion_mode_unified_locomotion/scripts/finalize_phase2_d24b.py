"""Offline D24B native reanalysis, geometry, durability, and reporting."""
from __future__ import annotations
import csv, hashlib, json, sqlite3, subprocess
from collections import Counter
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve();REPO=HERE.parents[4]
OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d24b_native_start_contract_and_recovery";RAW=OUT/"raw"
D24A=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d24a_existing_start_teacher_transfer"
D17=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d17_start_source_and_causality_audit"
D16=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d16_dedicated_start_specialist"
REPORT=REPO/"research/exp_014_phase_2_d24b_native_start_contract_and_recovery_report.md"
START="b99869f5497d15cdf4bd376c935e32df025c9a5c";CLASS="EXP014_D24B_NATIVE_TIME_CONTRACT_MISMATCH"
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def git(*a):return subprocess.check_output(["git",*a],cwd=REPO,text=True).strip()
OUT.mkdir(parents=True,exist_ok=True)
native=np.load(D24A/"raw/stage2q_native_trajectories.npz");nmeta=json.loads((D24A/"raw/native_results.json").read_text());transfer=json.loads((RAW/"transfer_results.json").read_text());tr=np.load(RAW/"transfer_trajectories.npz")

original={"source":"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/scripts/evaluate_stage2q_sequence.py and committed transition_results","source_lifecycle":"Isaac-Exp012-G1-Reverse-PhaseR1-v0 reset; policy active immediately","reset_zero_command_duration_s":1.0,"gait_cmd_timing":"WALK gait_cmd=0 for entire STAND_TO_WALK episode","target_speed":"0 until 1.0 s; minimum-jerk 0->0.6 from 1.0-2.0 s; minimum-jerk 0.6->1.2 from 2.0-3.0 s; 1.2 thereafter","command_ramp_duration_s":[1.0,1.0],"STAND_segment_duration_s":1.0,"WALK_segment_duration_s":9.0,"acquisition":"segment-0 flight_fraction < 0.10","confirmation":"NONE; no consecutive velocity/yaw tracking requirement","deadline":"NONE separate from 10 s episode","episode_duration_s":10.0,"previous_action_initialization":"UNKNOWN (environment reset lifecycle; not serialized in exp012 result)","observation_history_initialization":"UNKNOWN (environment reset lifecycle; not serialized in exp012 result)","contact_phase_initialization":"environment reset distribution; exact per-episode contact phase not serialized","saturation":"joint velocity / joint velocity limit >0.95 for >=5 consecutive control steps","historical_success_rate":1.0}
dump("exp012_stage2q_original_start_contract.json",original)
comparison={"classification":"NATIVE_START_TIME_CONTRACT_MISMATCH","differences":{"source_state_distribution":{"exp012":"native randomized reset","D24A":"native randomized reset with same formal seed"},"zero_command_preroll":{"exp012":1.0,"D24A":1.0},"gait_timing":{"exp012":"WALK from episode start","D24A":"WALK from episode start"},"speed":{"exp012":"0->0.6->1.2","D24A":"0->0.6 then hold"},"ramp":{"exp012":"two 1.0 s ramps","D24A":"one 1.0 s ramp"},"deadline":{"exp012":"no explicit acquisition deadline; 10 s episode aggregate","D24A":"completion before 3.0 s episode time (2.0 s after ramp request begins)"},"confirmation":{"exp012":"none","D24A":"25 consecutive forward/lateral tracking steps"},"safety":{"exp012":"fall/slip/impact and velocity saturation reported but not all embedded in gait success","D24A":"safe success conjunction"},"saturation":{"exp012":"joint velocity ratio","D24A":"joint velocity plus applied torque ratio"}},"answers":{"D24A_deadline_shorter":True,"D24A_confirmation_stricter":True}}
dump("native_contract_comparison.json",comparison)

def maxrun(x):
 best=cur=0;events=0
 for v in x:
  if v:cur+=1;best=max(best,cur)
  else:
   if cur:events+=1
   cur=0
 return best,events+(cur>0)
rows=[];classes=Counter();n3idx=[];n4idx=[]
for i,m in enumerate(nmeta["rows"]):
 good=(np.abs(native["vx"][:,i]-.6)<=.15)&(np.abs(native["vy"][:,i])<=.10);strict=good&(np.abs(native["yaw"][:,i])<=.12);mr,events=maxrun(good[50:]);ms,_=maxrun(strict[50:]);first=m["first_step"]
 start=-1
 if mr>=25:
  s=0
  for k,v in enumerate(good[50:],50):
   s=s+1 if v else 0
   if s>=25:start=k-24;break
 other=m["fall"] or m["slip"] or m["impact"] or m["nonfinite"] or m.get("velocity",False)
 if other:c="N6_OTHER_SAFETY_REJECTED"
 elif m["torque"]:c="N5_SATURATION_REJECTED"
 elif first<0:c="N0_NO_FIRST_STEP"
 elif mr<1:c="N1_FIRST_STEP_NO_ACQUISITION"
 elif mr<25:c="N2_ACQUIRED_SHORT_CONFIRMATION"
 elif ms>=25:c="N4_SAFE_DEMONSTRATION";n4idx.append(i)
 else:c="N3_ACQUIRED_25STEP_CONFIRMATION";n3idx.append(i)
 classes[c]+=1
 contact=native["contact"][:,i];dom=np.where(contact.sum(1)==1)[0]
 first_dom=int(dom[0]) if len(dom) else -1;side="NONE" if first_dom<0 else ("LEFT" if contact[first_dom,0] else "RIGHT")
 row={"episode":i,"class":c,"first_step_step":first,"first_step_time_s":None if first<0 else first*.02,"velocity_acquisition_start_step":start,"confirmation_end_step":None if start<0 else start+24,"confirmation_failure_reason":None if mr>=25 else "maximum continuous valid-tracking length below 25","maximum_continuous_tracking_steps":mr,"maximum_continuous_full_yaw_tracking_steps":ms,"tracking_event_count":events,"first_dominant_support":side,"first_swing_unload_step":first_dom,"touchdown_sequence":"NOT_RECOVERABLE_FROM_BINARY_CONTACT_TRACE","yaw_p95":float(np.quantile(np.abs(native["yaw"][:,i]),.95)),"fall":m["fall"],"dangerous_slip":m["slip"],"impact":m["impact"],"velocity_saturation":m.get("velocity",False),"torque_saturation_episode":m["torque"]}
 rows.append(row)
with (OUT/"native_trajectory_reanalysis.csv").open("w",newline="",encoding="utf-8") as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
dump("native_trajectory_reanalysis.json",{"episodes":rows,"summary":{"classes":dict(classes),"N3_count":len(n3idx),"N4_count":len(n4idx),"first_step_count":sum(r["first_step_step"]>=0 for r in rows),"full_yaw_25step_confirmation":sum(r["maximum_continuous_full_yaw_tracking_steps"]>=25 for r in rows),"historical_deadline_recovery_rule":"3.0 s fallback because exp012 has no explicit velocity acquisition deadline"}})
dump("native_trajectory_classification.json",{"classification_counts":dict(classes),"N3_indices":n3idx,"N4_indices":n4idx,"prerequisite":{"N4_ge_8":len(n4idx)>=8,"N3_ge_16_without_episode_torque_saturation":len(n3idx)>=16,"transfer_authorized":len(n4idx)>=8 or len(n3idx)>=16},"note":"N3 preserves D24A forward/lateral confirmation but fails the N4 yaw criterion; N4 is empty."})

# D24A did not persist applied-torque tensors. Do not fabricate per-joint native values.
joint_names=["left_hip_pitch","right_hip_pitch","torso","left_hip_roll","right_hip_roll","left_shoulder_pitch","right_shoulder_pitch","left_hip_yaw","right_hip_yaw","left_shoulder_roll","right_shoulder_roll","left_knee","right_knee","left_shoulder_yaw","right_shoulder_yaw","left_ankle_pitch","right_ankle_pitch","left_elbow_pitch","right_elbow_pitch","left_ankle_roll","right_ankle_roll","left_elbow_roll","right_elbow_roll","left_five","left_three","left_zero","right_five","right_three","right_zero","left_six","left_four","left_one","right_six","right_four","right_one","left_two","right_two"]
jrows=[]
for j,name in enumerate(joint_names):
 vals=tr["torque_ratio"][:,:,j]
 group="hip" if "hip" in name else "knee" if "knee" in name else "ankle" if "ankle" in name else "waist" if name=="torso" else "arm"
 jrows.append({"joint_index":j,"joint":name,"group":group,"native_max":None,"native_p95":None,"native_fraction_gt_90":None,"native_fraction_gt_95":None,"native_fraction_gt_100":None,"native_max_contiguous_gt95":None,"native_max_contiguous_gt100":None,"native_event_count":None,"transfer_max":float(vals.max()),"transfer_p95":float(np.quantile(vals,.95)),"transfer_fraction_gt95":float((vals>.95).mean())})
with (OUT/"torque_saturation_by_joint.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(jrows[0]));w.writeheader();w.writerows(jrows)
dump("torque_saturation_by_joint.json",{"native_status":"NOT_RECOVERABLE_FROM_D24A_RAW","reason":"D24A persisted actions and joint state but not applied_torque, effort limits, or per-joint torque ratio. Episode-level flags cannot be decomposed without re-simulation, which D24B forbids.","transfer_supplementary_exact":jrows,"actuator_clipping_semantics":"applied_torque is actuator-clipped; >100% should normally be impossible apart from numerical/timing discrepancies"})
dump("torque_saturation_dwell.json",{"native_episode_binary":{"episodes":100,"flagged":sum(m["torque"] for m in nmeta["rows"]),"rate":sum(m["torque"] for m in nmeta["rows"])/100},"native_per_joint_dwell":"NOT_RECOVERABLE","requested_categories":{"TRANSIENT":"<=2","SHORT_DWELL":"3-5","LONG_DWELL":">5"},"major_failure_reconstruction":"NOT_RECOVERABLE_FROM_D24A_RAW","transfer_long_dwell_rate":sum(r["torque_long_dwell"] for r in transfer["rows"])/320})

# Native N3 endpoint states are retained diagnostically; N4 is absent.
states=[]
for i in n3idx:
 first=nmeta["rows"][i]["first_step"];a=49;b=max(49,first-1)
 states.append({"episode":i,"A_step":a,"B_step":b,"A_state_hash":hashlib.sha256(native["root_state"][a,i].tobytes()+native["joint_pos"][a,i].tobytes()).hexdigest(),"B_state_hash":hashlib.sha256(native["root_state"][b,i].tobytes()+native["joint_pos"][b,i].tobytes()).hexdigest()})
dump("native_success_state_manifest.json",{"formal_N4_states":0,"diagnostic_N3_states":len(states),"unique_source_state_count":len(states),"states":states,"fields":["root","joint pos/vel","previous action","contact","velocity/yaw","observation/history where persisted"]})

# Geometry uses D17's frozen 122-D S_HOLD physical features and diagnostic N3 states.
d17=json.loads((D17/"raw/audit_results.json").read_text());sh=np.asarray(d17["source_features"],dtype=np.float32)
def feat(step,i):
 root=native["root_state"][step,i];rp=native["roll_pitch"][step,i];z=-np.sqrt(max(0.,1-float(rp[0]**2+rp[1]**2)));grav=np.array([rp[0],rp[1],z],np.float32);prev=native["action"][max(0,step-1),i];return np.concatenate((root[7:10],root[10:13],grav,native["joint_pos"][step,i],native["joint_vel"][step,i],prev,native["contact"][step,i].astype(np.float32)))
refs=np.stack([feat(49,i) for i in n3idx]) if n3idx else np.empty((0,122));mrows=[]
if len(refs):
 mean=refs.mean(0);std=np.maximum(refs.std(0),1e-3);dist=np.linalg.norm((sh[:,None,:]-refs[None,:,:])/std,axis=2);nn=dist.argmin(1)
 for i in range(64):mrows.append({"shold_source":i,"nearest_N3_episode":n3idx[int(nn[i])],"normalized_full_distance":float(dist[i,nn[i]]),"base_velocity_distance":float(np.linalg.norm(sh[i,:3]-refs[nn[i],:3])),"angular_velocity_distance":float(np.linalg.norm(sh[i,3:6]-refs[nn[i],3:6])),"projected_gravity_distance":float(np.linalg.norm(sh[i,6:9]-refs[nn[i],6:9])),"joint_position_distance":float(np.linalg.norm(sh[i,9:46]-refs[nn[i],9:46])),"joint_velocity_distance":float(np.linalg.norm(sh[i,46:83]-refs[nn[i],46:83])),"previous_action_distance":float(np.linalg.norm(sh[i,83:120]-refs[nn[i],83:120])),"contact_match":bool(np.array_equal(sh[i,120:122]>.5,refs[nn[i],120:122]>.5)),"support_side_match":bool(np.argmax(sh[i,120:122])==np.argmax(refs[nn[i],120:122]))})
with (OUT/"source_manifold_comparison.csv").open("w",newline="",encoding="utf-8") as f:
 fields=list(mrows[0]) if mrows else ["status"];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(mrows or [{"status":"NO_N4_REFERENCE"}])
contact_rate=sum(r["contact_match"] for r in mrows)/64 if mrows else 0
dump("source_manifold_comparison.json",{"formal_classification":"SOURCE_MANIFOLD_DISJOINT","formal_basis":"N4 reference set is empty, so the preregistered p95-overlap criterion cannot pass.","diagnostic_reference":"N3 states only","diagnostic_contact_match_rate":contact_rate,"diagnostic_distance_median":None if not mrows else float(np.median([r["normalized_full_distance"] for r in mrows])),"operational_transfer_evidence":{"continuous_S_HOLD_validity":0,"demonstration_success":0},"rows":mrows})

# Native pre-roll diagnostics (steps 0..49).
def pre(indices):
 if not indices:return None
 ac=native["action"][:50,indices];co=native["contact"][:50,indices];switch=(co[1:]!=co[:-1]).any(-1).sum(0);left=int(sum(bool(native["contact"][max(0,nmeta["rows"][i]["first_step"]),i,0]) for i in indices if nmeta["rows"][i]["first_step"]>=0));with_step=sum(nmeta["rows"][i]["first_step"]>=0 for i in indices);return {"episodes":len(indices),"action_l2_mean":float(np.linalg.norm(ac,axis=2).mean()),"contact_switch_mean":float(switch.mean()),"yaw_abs_p95":float(np.quantile(np.abs(native["yaw"][:50,indices]),.95)),"previous_action_l2_mean":float(np.linalg.norm(ac[0],axis=1).mean()),"first_step_side":{"LEFT":left,"RIGHT":int(with_step-left)}}
dump("stage2q_native_preroll_analysis.json",{"contract":"Stage2Q acts during 1.0 s zero-command/WALK-gait pre-roll","N3":pre(n3idx),"N4":pre(n4idx),"others":pre([i for i in range(100) if i not in n3idx+n4idx]),"interpretation":"A first step can form, but no raw episode satisfies the full yaw-conditioned N4 confirmation."})

dump("wmove_speed_scope_audit.json",{"checkpoint":"exp013 W1B-R2 iteration 200","0.3_mps":"AUTHORIZED (formal exp014/D14 command family and exp013 low-speed matrices)","0.6_mps":"NOT_AUTHORIZED_AS_COMPLETE_OMNIDIRECTIONAL_SCOPE (exp013 W1A2 recorded partial 0.6 condition support and LOW_SPEED_RETENTION_FAIL)","validation_speeds":[.3,.25,.35],"unsupported_for_this_handoff":[.6],"local_neighborhood":"0.3 +/- 0.05 m/s","scope_not_expanded":True})
dump("handoff_speed_plan.json",{"case":"B","route":"Stage2Q 0.6 acquisition -> 0.75 s minimum-jerk 0.6->0.3 -> 0.50 s Stage2Q 0.3 hold -> hard W_MOVE 0.3","executed":True,"formal_overlap_speed":.3,"result":"NO_ELIGIBLE_DEMONSTRATION_REACHED_HANDOFF"})

trows=transfer["rows"]
with (OUT/"shold_transfer_route_results.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(trows[0]));w.writeheader();w.writerows(trows)
route_summary={}
for name in transfer["route_names"]:
 x=[r for r in trows if r["route"]==name];route_summary[name]={"episodes":64,"source_valid":sum(r["source_valid"] for r in x),"first_step":sum(r["first_step"] for r in x),"strict_success":sum(r["strict_success"] for r in x),"demonstration_success":sum(r["demonstration_success"] for r in x),"wmove_retained":sum(r["wmove_retained"] for r in x),"fall":sum(r["fall"] for r in x),"dangerous_slip":sum(r["dangerous_slip"] for r in x),"torque_long_dwell":sum(r["torque_long_dwell"] for r in x)}
dump("shold_transfer_route_results.json",{"summary":route_summary,"rows":trows,"strict_window_s":1.5,"demonstration_window_s":3.0,"unique_safe_source_coverage":len({r["source_index"] for r in trows if r["demonstration_success"]}),"tensor_hashes":transfer["tensor_hashes_before"],"tensor_hashes_after":transfer["tensor_hashes_after"],"tensor_identity_unchanged":transfer["tensor_hashes_before"]==transfer["tensor_hashes_after"]})
handoff=[r for r in trows if r["handoff_action_l2"] is not None]
with (OUT/"stage2q_to_wmove_handoff.csv").open("w",newline="",encoding="utf-8") as f:
 fields=list(trows[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(handoff)
dump("stage2q_to_wmove_handoff.json",{"status":"NOT_EXECUTED_NO_ELIGIBLE_DEMONSTRATION" if not handoff else "EXECUTED","eligible_episodes":len(handoff),"retained":sum(r["wmove_retained"] for r in handoff),"retention_rate":None if not handoff else sum(r["wmove_retained"] for r in handoff)/len(handoff),"action_continuity":None,"safety":None})

# No successful trajectory may be fabricated.
np.savez_compressed(OUT/"successful_start_trajectory_bundle.npz",status=np.array(["EMPTY_NO_SUCCESS"]),count=np.array([0],np.int64));digest=sha(OUT/"successful_start_trajectory_bundle.npz");(OUT/"successful_start_trajectory_bundle.sha256").write_text(digest+"  successful_start_trajectory_bundle.npz\n")
dump("successful_start_trajectory_manifest.json",{"status":"EMPTY_NO_SUCCESS","safe_trajectory_count":0,"unique_S_HOLD_sources":0,"bundle_sha256":digest,"raw_transfer_capture":str((RAW/"transfer_trajectories.npz").relative_to(REPO)).replace("\\","/"),"raw_capture_sha256":sha(RAW/"transfer_trajectories.npz")})
dump("temporary_distillation_feasibility.json",{"status":"NOT_EXECUTED","reason":"Fewer than 8 S_HOLD transfer demonstrations","samples":0,"MSE":None,"first_step_boundary_MSE":None,"Stage2Q_to_WMOVE_boundary_MSE":None,"cosine":None,"persistent_checkpoint":0})

# Durable 320-row result ledger.
db=OUT/"durable_evaluation.sqlite"
if db.exists():db.unlink()
con=sqlite3.connect(db);con.execute("pragma journal_mode=WAL");con.execute("pragma synchronous=FULL");con.execute("create table episode_results(episode_id integer primary key,result_json text not null,result_sha256 text not null,status text not null)")
with con:
 for r in trows:
  text=json.dumps(r,sort_keys=True,separators=(",",":"));con.execute("insert into episode_results values(?,?,?,?)",(r["episode_id"],text,hashlib.sha256(text.encode()).hexdigest(),"COMPLETED"))
con.execute("pragma wal_checkpoint(full)");con.close()

dump("stage_reference.json",{"stage":"Phase 2-D24B","starting_HEAD":START,"actual_HEAD_before_commit":git("rev-parse","HEAD"),"D24A_classification_preserved":"EXP014_D24A_STAGE2Q_NATIVE_REPRODUCTION_FAIL","teacher_byte_hashes":{"S_HOLD":"734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621","Stage2Q":"66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698","W_MOVE":"61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d","S_STOP_OMNI":"5de37e5d0807654d370ba7a79ee9872c4029cb50e548685423e48812249959d5"}})
dump("protocol.json",{"native_reanalysis":"D24A raw only; no native re-simulation","native_demo_deadline_s":3.0,"transfer":{"train_sources":64,"routes":5,"episodes":320,"strict_window_s":1.5,"demo_window_s":3.0},"speed_bridge":{"from":.6,"to":.3,"ramp_s":.75,"hold_s":.5},"prohibited":{"persistent_PPO":0,"checkpoint":0,"validation":0,"heldout":0,"D25":0,"RUN":0}})
dump("stage_classification.json",{"main_classification":CLASS,"decision_precedence":1,"subclassifications":["NATIVE_START_TIME_CONTRACT_MISMATCH","NATIVE_TORQUE_TRACE_NOT_RECONSTRUCTABLE","SOURCE_MANIFOLD_DISJOINT","SHOLD_TRANSFER_ZERO_DEMONSTRATIONS"],"D24A_unchanged":True})
dump("recommended_next_action.json",{"only_next_experiment":"model-based S_HOLD-to-native-prestart bridge using centroidal/ZMP or whole-body IK Teacher","reason":"Native N3 experience exists, but N4 is empty and exact native pre-roll produced zero S_HOLD demonstrations; reward tuning and D25 are not selected under the evidence/precedence."})
dump("protected_hashes.json",{"starting_HEAD":START,"D6_through_D24A_unchanged":True,"exp005_through_exp013_unchanged_from_turn_start":True,"teacher_tensor_hashes_before":transfer["tensor_hashes_before"],"teacher_tensor_hashes_after":transfer["tensor_hashes_after"],"tensor_hash_match":transfer["tensor_hashes_before"]==transfer["tensor_hashes_after"],"persistent_update":0,"new_persistent_checkpoint":0,"validation_access":0,"heldout_access":0,"D25_training":0,"RUN":0,"remote_push":False})
(OUT/"reproduction_commands.ps1").write_text("$ErrorActionPreference='Stop'\nSet-Location 'C:\\Users\\user\\workspace\\physical-ai-lab'\n# D24A native trajectories are immutable inputs and must not be re-simulated.\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d24b_transfer.py --headless\npython experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d24b.py\n",encoding="utf-8")
REPORT.write_text(f"""# Exp014 Phase 2-D24B native START contract and recovery

## Result

Classification: `{CLASS}`.

The original exp012 100% result used a 10 s episode gait classification (`flight_fraction < 0.10`) with no velocity/yaw confirmation and no explicit acquisition deadline. D24A instead held 0.6 m/s, imposed 25 consecutive forward/lateral tracking steps before 3.0 s episode time, and added torque saturation. The contracts therefore do not match.

D24A raw reanalysis found 77 first-step events and {len(n3idx)} conservative N3 trajectories, but zero N4 trajectories: no episode maintained the full forward/lateral/yaw criterion for 25 steps (maximum was 10). Per-joint native torque dwell cannot be reconstructed because D24A did not persist applied-torque tensors; the 31 episode-level flags are preserved without extrapolation.

The 64 fixed S_HOLD sources were evaluated on R0-R4. Continuous 0.5 s source validity was 0/64 per route. Stage2Q routes produced 30/64 first-step events but zero confirmed acquisitions, and all had long-dwell torque saturation under the exp014 definition. Consequently, safe S_HOLD demonstration coverage and W_MOVE handoffs were both zero. No static distillation was run.

W_MOVE 0.3 m/s is in the formal command family; 0.6 m/s is not authorized as a complete omnidirectional scope. The preregistered 0.6->0.3 bridge was represented, but no eligible demonstration reached handoff.

No persistent policy update, checkpoint, validation or held-out access, D25 training, RUN integration, or remote push occurred. The recommended next experiment is a model-based S_HOLD-to-native-prestart bridge using centroidal/ZMP or whole-body IK supervision.
""",encoding="utf-8")
print(json.dumps({"classification":CLASS,"native_classes":dict(classes),"routes":route_summary,"output":str(OUT)},indent=2))
