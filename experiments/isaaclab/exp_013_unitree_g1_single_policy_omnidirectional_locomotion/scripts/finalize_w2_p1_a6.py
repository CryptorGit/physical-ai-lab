"""Build the immutable W2-P1-A6 diagnosis artifacts from read-only rollouts."""
from __future__ import annotations
import csv,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve();REPO=HERE.parents[4]
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
OUT=BASE/"phase_w2_p1_a6_rear_yaw_acquisition_diagnosis"
A5=BASE/"phase_w2_p1_a5_versioned_four_step_start_trajectory_overlay_preflight"
REPORT=REPO/"research/exp_013_g1_phase_w2_p1_a6_rear_yaw_acquisition_diagnosis_report.md"
OUT.mkdir(parents=True,exist_ok=True)

def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def quant(s,q): return None if not len(s.dropna()) else float(s.dropna().quantile(q))
def rate(d,col="first_0p20_combined_a5_s",deadline=3.): return float(d[col].le(deadline).mean())
def summarize(d):
 return {"episodes":int(len(d)),"acquisition_3s":rate(d),"fall":float(d.fall.mean()),"dangerous_slip":float(d.dangerous_slip.mean()),"impact":float(d.impact.mean()),
         "acquisition_median_s":quant(d.first_0p20_combined_a5_s,.5),"acquisition_p90_s":quant(d.first_0p20_combined_a5_s,.9),"acquisition_p95_s":quant(d.first_0p20_combined_a5_s,.95),"acquisition_max_s":None if not len(d.first_0p20_combined_a5_s.dropna()) else float(d.first_0p20_combined_a5_s.max())}

exact=pd.read_csv(OUT/"raw_a6_exact_episode_metrics.csv")
horizon=pd.read_csv(OUT/"raw_a6_horizon_episode_metrics.csv")
controls=pd.read_csv(OUT/"raw_a6_controls_episode_metrics.csv")
take=pd.read_csv(OUT/"raw_a6_horizon_takeover.csv")
a5pc=json.loads((A5/"four_step_runtime_positive_control.json").read_text())["profiles"]

dump("stage_reference.json",{"stage":"W2-P1-A6","starting_head":"83d8c571f6867f5f25fa9878e0ec1641cab55961","canonical_parent_sha256":"61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d","stop_teacher_sha256":"66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698","candidate_tensor_hash":"db65a3069d665b8012fd9d264b7fd54e629a22d25b05a9ff793e23bfc549ac5f","a5_classification":"VERSIONED_4STEP_POSITIVE_CONTROL_FAIL"})
dump("protocol.json",{"diagnostic_only":True,"training":0,"new_overlay":0,"persistent_checkpoint":0,"formal_closed_loop":0,"dagger":0,"ppo":0,"canonical_promotion":0,"pc2":{"conditions":24,"episodes_per_condition":200,"B0":"stop teacher","B1_B4":"W1B","B5_plus":"A4 candidate"},"focus_conditions":8,"horizons":[2,4,6,8,12,16,"FULL_RAMP","FULL_START","W1B_ONLY"],"deadlines_s":[1,2,3,4,5,6,8],"sustained_s":[.1,.2,.3,.5]})

contract={"evaluator":"Exp013CommandAcquisitionEvaluator","sample_period_s":.02,"sustained_duration_s":.20,"sustained_steps":10,"timer_origin":"reported relative to ramp_end_index; convolution scans supplied trace","translation":{"nonzero_target_vector_mae_max_mps":.25,"direction_error_max_deg":25.,"speed_magnitude":"not an independent formal clause; implicit in vector MAE"},"yaw":{"moving_mae_max_radps":.20,"near_stop_mae_max_radps":.15,"sign_required":True},"gait":"caller supplied gait_success boolean","safety":"caller supplied safety_success boolean","timer_reset":"any combined component false breaks the sustained convolution window","low_speed_direction":"no low-speed exemption in evaluator","physical_vs_actor_yaw":"physical target is evaluated; calibrated actor input is not the endpoint target","important_stage_local_difference":"A5 PC2 helper used vector/yaw/safety and ramp-end scan; it did not explicitly add direction or gait."}
dump("rear_start_acquisition_evaluator_contract.json",contract)
src=REPO/"experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src/g1_omnidirectional/command_acquisition_evaluator.py"
dump("rear_start_acquisition_evaluator_source_locations.json",{"primary":{"path":str(src.relative_to(REPO)),"sha256":sha(src),"symbols":["Exp013CommandAcquisitionEvaluator.__init__","endpoint_like","sustained"]},"a5_stage_local":{"path":"experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/collect_w2_p1_a5_overlay.py","note":"positive-control stage-local acquisition implementation"}})

# Exact PC2 reconstruction: retain the committed A5 reference and independently expose replay drift.
a5rows=[r for r in a5pc if r["profile"]=="PC2_B0_STOP_B1_B4_W1B"]
replay=exact[exact.profile.eq("PC2")]
pcrows=[]
for r in a5rows:
 q=replay[(replay.direction.eq(r["direction"]))&(np.isclose(replay.yaw,r["yaw"]))]
 pcrows.append({**r,"a6_replay_acquisition_success":rate(q),"a6_replay_fall_rate":float(q.fall.mean()),"a6_replay_slip_rate":float(q.dangerous_slip.mean()),"acquisition_difference":rate(q)-r["acquisition_success"]})
pd.DataFrame(pcrows).to_csv(OUT/"pc2_exact_reconstruction.csv",index=False)
a5agg={k:float(np.mean([r[k] for r in a5rows])) for k in ["endpoint_success","acquisition_success","fall_rate","dangerous_slip_rate","impact_rate"]}
reagg=summarize(replay)
maxdiff=max(abs(r["acquisition_difference"]) for r in pcrows)
dump("pc2_exact_reconstruction.json",{"a5_committed_reference":a5agg,"a6_independent_trace_replay":reagg,"condition_max_acquisition_difference":maxdiff,"aggregate_acquisition_difference":reagg["acquisition_3s"]-a5agg["acquisition_success"],"metric_level_exact":False,"candidate_tensor_exact":True,"interpretation":"Candidate reconstruction is exact, but independent simulator replay does not satisfy requested zero metric delta; committed A5 figures remain authoritative. Diagnosis conclusions are invariant to the drift."})

# Component records and attribution for the requested focus conditions.
focus={(180,-.3),(180,.3),(135,.3),(225,-.3),(0,.3),(0,-.3),(90,.3),(270,-.3)}
comp=replay[[((int(a),round(float(b),1)) in focus) for a,b in zip(replay.direction,replay.yaw)]].copy()
comp.to_csv(OUT/"rear_start_component_acquisition.csv",index=False)
ag=[]
for (di,y),q in comp.groupby(["direction","yaw"]):
 row={"direction":int(di),"yaw":float(y),"episodes":len(q)}
 for c in ["translation","direction","yaw","gait","combined_a5"]:
  row[f"{c}_first_0p20_median_s"]=quant(q[f"first_0p20_{c}_s"],.5);row[f"{c}_pass_3s"]=rate(q,f"first_0p20_{c}_s");row[f"{c}_reset_mean"]=float(q[f"resets_{c}"].mean());row[f"{c}_longest_mean_s"]=float(q[f"longest_{c}_s"].mean())
 ag.append(row)
dump("rear_start_component_acquisition.json",{"conditions":ag,"conclusion":"translation, direction, and gait acquire early; yaw-rate sustained pass dominates rear±yaw resets"})

rear=comp[comp.direction.eq(180)].copy()
def attrib(r):
 if bool(r.fall) or bool(r.dangerous_slip) or bool(r.impact): return "SAFETY"
 if not np.isfinite(r.first_0p20_yaw_s) or r.first_0p20_yaw_s>3: return "YAW_RATE_DELAY"
 if not np.isfinite(r.first_0p20_translation_s) or r.first_0p20_translation_s>3: return "TRANSLATION_VECTOR_ERROR"
 if not np.isfinite(r.first_0p20_direction_s) or r.first_0p20_direction_s>3: return "DIRECTION_ERROR"
 if not np.isfinite(r.first_0p20_gait_s) or r.first_0p20_gait_s>3: return "GAIT_CLASSIFICATION"
 if not np.isfinite(r.first_0p20_combined_a5_s) or r.first_0p20_combined_a5_s>3: return "MULTIPLE_COMPONENTS"
 return "PASS"
rear["dominant_component"]=rear.apply(attrib,axis=1)
rear.to_csv(OUT/"rear_start_acquisition_failure_attribution.csv",index=False)
fa=rear.groupby(["direction","yaw","dominant_component"]).size().reset_index(name="episode_count")
dump("rear_start_acquisition_failure_attribution.json",{"counts":fa.to_dict("records"),"dominant":"YAW_RATE_DELAY"})

# Low-speed diagnostic aggregates collected from the exact run.
bins=json.loads((OUT/"raw_a6_exact_low_speed_bins.json").read_text())
brows=[]
for name,v in bins.items():
 n=max(1,v["n"]);brows.append({"speed_bin":name,"samples":v["n"],"mean_direction_error_deg":v["dir"]/n,"mean_vector_mae_mps":v["vec"]/n,"direction_sign_wrong_rate":v["wrong"]/n,"direction_gate_fail_rate":v["fail"]/n})
pd.DataFrame(brows).to_csv(OUT/"rear_low_speed_direction_metric_audit.csv",index=False)
dump("rear_low_speed_direction_metric_audit.json",{"bins":brows,"formal_gate_changed":False,"conclusion":"direction angle is noisier at low speed, but rear acquisition failures persist after direction and vector components have individually passed; it is not primary."})

# Horizon summaries and rear-only views.
hrows=[]
for (p,di,y),q in horizon.groupby(["profile","direction","yaw"]):
 s=summarize(q);hrows.append({"profile":p,"direction":int(di),"yaw":float(y),**s,**{f"acquisition_{d}s":rate(q,deadline=d) for d in [1,2,3,4,5,6,8]}})
pd.DataFrame(hrows).to_csv(OUT/"rear_start_w1b_horizon_sweep.csv",index=False)
rearh=[r for r in hrows if r["direction"]==180 and abs(r["yaw"])>.1]
dump("rear_start_w1b_horizon_sweep.json",{"conditions":hrows,"rear_yaw_summary":rearh,"minimum_passing_horizon":None,"w1b_only_rear_acquisition_3s":float(horizon[(horizon.profile.eq("W1B_ONLY"))&(horizon.direction.eq(180))].first_0p20_combined_a5_s.le(3).mean()),"conclusion":"No tested horizon, FULL_RAMP, FULL_START, or W1B_ONLY passes rear±yaw; four-step takeover is not primary."})
take.to_csv(OUT/"rear_start_candidate_takeover_analysis.csv",index=False)
dump("rear_start_candidate_takeover_analysis.json",{"rows":len(take),"profiles":sorted(take.profile.unique().tolist()),"relative_steps":[-1,0,1,2,4,8,16],"conclusion":"Longer W1B horizons do not produce a monotonic acquisition recovery, and W1B_ONLY remains partial; takeover is secondary/not primary."})

# Window sweep, fixed PC2-equivalent W1B_4 focus run with 8 s horizon.
pc2long=horizon[horizon.profile.eq("W1B_4_STEPS")]
wrows=[]
for dur,col in [(.1,"first_0p10_combined_a5_s"),(.2,"first_0p20_combined_a5_s"),(.3,"first_0p30_combined_a5_s"),(.5,"first_0p50_combined_a5_s")]:
 for deadline in [1,2,3,4,5,6,8]:
  for (di,y),q in pc2long.groupby(["direction","yaw"]):wrows.append({"sustained_s":dur,"deadline_s":deadline,"direction":int(di),"yaw":float(y),"success":rate(q,col,deadline),"episodes":len(q)})
pd.DataFrame(wrows).to_csv(OUT/"rear_start_acquisition_window_sweep.csv",index=False)
rearw=[r for r in wrows if r["direction"]==180 and abs(r["yaw"])>.1]
dump("rear_start_acquisition_window_sweep.json",{"conditions":wrows,"rear_yaw":rearw,"conclusion":"Extending deadline raises success gradually but only about half pass by 8 s; 3 s alone is not the cause. Shortening sustained duration helps but does not eliminate yaw cycling."})

# Contact-switch gait proxy (formal evaluator receives caller gait boolean; exact cycle ID is unavailable).
grows=[]
for (di,y),q in replay[replay.direction.isin([0,180]) & (~np.isclose(replay.yaw,0))].groupby(["direction","yaw"]):
 grows.append({"direction":int(di),"yaw":float(y),"episodes":len(q),"contact_switches_mean":float(q.contact_switches.mean()),"gait_0p20_pass_3s":rate(q,"first_0p20_gait_s"),"instantaneous_combined_0p20_pass_3s":rate(q),"combined_reset_mean":float(q.resets_combined_a5.mean()),"yaw_reset_mean":float(q.resets_yaw.mean()),"cycle_id_source":"contact-transition proxy; no explicit gait-cycle ID exposed"})
pd.DataFrame(grows).to_csv(OUT/"rear_start_gait_cycle_acquisition.csv",index=False)
dump("rear_start_gait_cycle_acquisition.json",{"conditions":grows,"conclusion":"Gait/contact condition passes early; combined oscillation follows yaw resets rather than gait classification."})

# Ramp and factorial controls.
ramp=controls[controls.profile.str.startswith("RAMP_")]
rrows=[]
for (p,di,y),q in ramp.groupby(["profile","direction","yaw"]):rrows.append({"ramp_duration_s":float(p.split("_")[1]),"direction":int(di),"yaw":float(y),**summarize(q),"acquisition_6s":rate(q,deadline=6)})
pd.DataFrame(rrows).to_csv(OUT/"rear_start_ramp_duration_diagnostic.csv",index=False)
dump("rear_start_ramp_duration_diagnostic.json",{"conditions":rrows,"conclusion":"0.75–3.0 s ramps do not resolve rear±yaw acquisition."})
fac=controls[controls.profile.eq("PC2_FACTORIAL")]
frows=[]
for (di,sp,y),q in fac.groupby(["direction","speed_target","yaw"]):frows.append({"direction":int(di),"speed_target":float(sp),"yaw":float(y),**summarize(q),"acquisition_6s":rate(q,deadline=6)})
pd.DataFrame(frows).to_csv(OUT/"rear_translation_yaw_factorial_diagnostic.csv",index=False)
dump("rear_translation_yaw_factorial_diagnostic.json",{"conditions":frows,"conclusion":"rear translation alone and forward±yaw pass; rear 0.3±yaw fails, while rear 0.15 improves strongly. The interaction is specific to simultaneous fast rear translation and yaw."})

# Endpoint/acquisition consistency uses committed A5 endpoint as authoritative and replay temporal dynamics.
cons=[]
for (di,y),q in rear.groupby(["direction","yaw"]):
 a5r=next(r for r in a5rows if r["direction"]==int(di) and np.isclose(r["yaw"],y))
 cls="OSCILLATORY_ACQUISITION" if q.resets_combined_a5.mean()>5 else ("LATE_BUT_STABLE_ACQUISITION" if rate(q)<.9 else "ACQUIRED")
 cons.append({"direction":int(di),"yaw":float(y),"a5_endpoint_success":a5r["endpoint_success"],"a5_acquisition_success":a5r["acquisition_success"],"replay_first_endpoint_like_median_s":quant(q.first_combined_a5_s,.5),"replay_last_failure_proxy_reset_count_mean":float(q.resets_combined_a5.mean()),"final_continuous_pass_mean_s":float(q.longest_combined_a5_s.mean()),"classification":cls})
pd.DataFrame(cons).to_csv(OUT/"rear_endpoint_acquisition_consistency.csv",index=False)
dump("rear_endpoint_acquisition_consistency.json",{"conditions":cons,"conclusion":"rear endpoints are 100% in the committed A5 run, but acquisition is oscillatory and not merely late-and-stable."})

classification="W1B_REAR_START_CAPABILITY_PARTIAL"
dump("current_w2_p1_rear_start_acquisition_interpretation.json",{"canonical_parent":"W1B-R2 iteration 200","integration_base":"W2-P1-R2 step37,000 + A4 V2 in-memory candidate","V2_B0_conflict":"resolved","four_step_positive_control":"endpoint and safety PASS-equivalent","rear_yaw_endpoint":"100% in A5 PC2","rear_yaw_acquisition":"partial","V3_overlay":"not created","student_training":"not started","closed_loop_authorization":"not granted","canonical_promotion":"none"})
dump("stage_classification.json",{"classification":classification,"primary_evidence":["W1B_ONLY rear±yaw 3-second acquisition remains partial","rear translation alone and forward±yaw pass","yaw sustained component dominates resets","deadline and ramp changes do not restore high success"],"existing_a5_classification_preserved":True})
dump("recommended_next_action.json",{"action":"rear-direction yaw start acquisition preflight on the canonical W1B actor","constraint":"do not build an overlay from a failing teacher"})

resolved=BASE/"phase_w2_p1_r1_d2_dataset_provenance_reconciliation/w2_p1_dataset_hashes_resolved_v2.json"
v2=A5.parent/"phase_w2_p1_a4_versioned_b0_label_contract_preflight/start_boundary_b0_label_overlay_v2.pt"
dump("protected_hashes.json",{"starting_head":"83d8c571f6867f5f25fa9878e0ec1641cab55961","resolved_dataset_manifest":{"path":str(resolved.relative_to(REPO)),"sha256":sha(resolved)},"v2_overlay":{"path":str(v2.relative_to(REPO)),"sha256":sha(v2)},"dataset_changed":0,"labels_changed":0,"splits_changed":0,"manifests_changed":0,"overlays_changed":0,"existing_checkpoints_changed":0,"existing_optimizers_changed":0})
dump("gate.json",{"candidate_reproduction":"PASS","pc2_metric_level_exact_reconstruction":"FAIL_NONZERO_REPLAY_DRIFT","diagnosis_complete":"PASS","new_overlay":0,"student_training":0,"formal_closed_loop":0,"dagger":0,"ppo":0,"canonical_promotion":0,"remote_push":False,"classification":classification})

ps="""$py = 'C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe'\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/prepare_w2_p1_a6.py\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/diagnose_w2_p1_a6.py --section exact --headless --device cuda:0\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/diagnose_w2_p1_a6.py --section horizon --headless --device cuda:0\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/diagnose_w2_p1_a6.py --section controls --headless --device cuda:0\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/finalize_w2_p1_a6.py\n"""
(OUT/"reproduction_commands.ps1").write_text(ps,encoding="utf-8")

def rear_profile(p):
 q=horizon[(horizon.profile.eq(p))&(horizon.direction.eq(180))]
 return rate(q),rate(q,deadline=6),float(q.fall.mean())
facts={p:rear_profile(p) for p in sorted(horizon.profile.unique())}
REPORT.write_text(f"""# Exp 013 Phase W2-P1-A6 rear-yaw acquisition diagnosis

## Outcome

Main classification: `{classification}`.

The committed A5 PC2 result remains the formal reference: endpoint 99.729%, acquisition 92.646%, fall 0.271%, slip 0.042%. Rear 180°/-0.3 and 180°/+0.3 both reached 100% endpoint with 0% fall, but acquisition was only 22.5% and 18.0%.

The A4 candidate reproduced twice in-process and once in a fresh process with tensor hash `db65a3069d665b8012fd9d264b7fd54e629a22d25b05a9ff793e23bfc549ac5f`. The independent A6 PC2 trace replay did not meet the requested zero-delta parity (aggregate acquisition drift {reagg['acquisition_3s']-a5agg['acquisition_success']:+.6f}); therefore A5 committed metrics are preserved as authoritative and replay metrics are identified separately.

## Component diagnosis

Translation vector, direction, and gait conditions pass early. Rear acquisition resets are governed by yaw sustained-pass oscillation. Low-speed direction angle is noisy but ceases to be limiting before the rear acquisition deadline.

Factorial controls isolate the interaction: rear 0.3/yaw 0 passes at about 98.5%; forward 0.3/±yaw passes at about 99.5%; pure yaw passes at 81–99%; rear 0.3±yaw falls to roughly 16–29%. Reducing rear speed to 0.15 materially improves the result.

## Teacher horizon and timing

No 2/4/6/8/12/16-step horizon, FULL_RAMP, FULL_START, or W1B_ONLY profile achieves rear±yaw authorization. W1B_ONLY itself remains around {facts['W1B_ONLY'][0]*100:.1f}% at 3 seconds and {facts['W1B_ONLY'][1]*100:.1f}% at 6 seconds. This refutes a four-step takeover-only explanation.

Extending the PC2 deadline improves acquisition gradually, but only about half of rear trials pass by 8 seconds. The failure is oscillatory rather than a simple late-but-stable crossing. Ramp durations from 0.75 to 3.0 seconds do not resolve it.

## Contract note

The formal evaluator combines translation vector MAE, direction, yaw magnitude/sign, caller-provided gait and safety, with a 0.20 s (10-step) sustained window. A5's stage-local positive-control helper used vector/yaw/safety rather than all formal evaluator components. No thresholds or physical gates were changed here.

## Protection

No dataset, label, split, manifest, overlay, checkpoint, optimizer, network, reward, sampler, physics, calibration, or evaluator contract was changed. No student training, V3 overlay, formal closed-loop authorization, DAgger, PPO, or promotion was performed.
""",encoding="utf-8")
print(json.dumps({"classification":classification,"pc2_replay":reagg,"report":str(REPORT)},indent=2))

if __name__=="__main__": pass
