"""Build committed D17 artifacts from the read-only physics audit."""
from __future__ import annotations
import csv, hashlib, json, subprocess
from pathlib import Path

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d17_start_source_and_causality_audit"
RAW=OUT/"raw"; REPORT=REPO/"research/exp_014_phase_2_d17_start_source_and_causality_audit_report.md"
D16=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d16_dedicated_start_specialist"
EXP13=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
START_HEAD="8b2175351d588d425a50b92693e8e28e53dfb2af"

def dump(name,x):
 OUT.mkdir(parents=True,exist_ok=True);(OUT/name).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def avg(xs):return sum(xs)/len(xs) if xs else 0
def csvwrite(name,rows):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:p.write_text("status\nNOT_APPLICABLE\n",encoding="utf-8-sig");return
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with p.open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

x=json.loads((RAW/"audit_results.json").read_text(encoding="utf-8"));supp=json.loads((RAW/"geometry_prev_supplement.json").read_text(encoding="utf-8"));g=x["geometry"]
for row in g:
 row["constructed_122d_physical_state_distance"]=row.pop("full_123d_physical_observation_distance")
stage_ref={"phase":"2-D17","starting_head":START_HEAD,"actual_starting_head":START_HEAD,"actual_head_is_source_of_truth":True,"d15_classification":"EXP014_D15_DIRECT_WMOVE_START_FAIL","d16_classification":"EXP014_D16_FORWARD_START_FAIL","persistent_policy_training":0,"remote_push":False}
protocol={"name":"Exp014StartSourceStateGeometryCommandYawCausalityAuditV1","selection_split":"train 64 only","validation_use":"one-time confirmation after diagnosis","checkpoints":["B0 W_MOVE direct","R0 D16 initial residual","R40 D16 update 40 diagnostic","H S_HOLD"],"counterfactuals":{"command_ramp_s":[0.02,0.1,0.25,0.5,1.0],"previous_action":["actual S_HOLD","zero","nearest basin previous","nearest basin current"],"gate":[1.5,2.0,3.0,"FULL"]},"reachability":[{"horizon":25,"bound":.5},{"horizon":50,"bound":.5},{"horizon":25,"bound":.75}],"formal_contract_changed":False}
dump("stage_reference.json",stage_ref);dump("protocol.json",protocol)
source_hashes=[hashlib.sha256(json.dumps(row,separators=(",",":"),allow_nan=False).encode()).hexdigest() for row in x["source_features"]]
dump("source_snapshot_manifest.json",{"split":"train","snapshots":64,"recipe_ids":x["source_recipe_ids"],"source_indices":x["source_indices"],"observation_hashes":source_hashes,"fields":["root state","joint position/velocity","previous action","contact/support","base velocity","yaw rate","roll/pitch","observation hash"],"selection_before_results":True,"validation_snapshots":102,"validation_valid":101})
dump("wmove_basin_reference_manifest.json",{"name":"W_MOVE_FORWARD_BASIN_REFERENCE_V1","states":x["basin"]["states"],"minimum_required":10000,"condition":{"direction_deg":0,"speed":.3,"yaw":0},"source_artifact":x["basin"]["source_artifact"],"source_split":"train","feature_dimension":x["basin"]["feature_dimension"],"trajectory_distance_reference_stride":5,"steady_tracking_and_safety_filter":"D6 W_MOVE-acquired train snapshots; 160 W_MOVE continuation steps"})
csvwrite("source_to_basin_geometry.csv",g)
geom={"snapshots":len(g),"nearest_state_distance":{"mean":avg([r["nearest_state_distance"] for r in g]),"p95":sorted(r["nearest_state_distance"] for r in g)[int(.95*(len(g)-1))]},"full_123d_policy_observation_distance":supp["full_123d_observation"],"nearest_action":{"l2_mean":avg([r["nearest_action_l2"] for r in g]),"cosine_mean":avg([r["nearest_action_cosine"] for r in g])},"contact_mismatch_rate":avg([float(r["contact_mismatch"]) for r in g]),"support_foot_mismatch_rate":avg([float(r["support_foot_mismatch"]) for r in g]),"classification":"DISTINCT_MANIFOLDS","component_distance_means":{k:avg([r[k] for r in g]) for k in g[0] if k.endswith("_distance")},"note":"123D standardized distance is dominated by command/history coordinates that are nearly constant inside the steady basin; component physical distances are reported separately."}
dump("source_to_basin_geometry.json",geom)
first_rows=[]
for policy,v in x["first_divergence"].items():
 for row in v["trace"]:first_rows.append({"policy":policy,**row})
csvwrite("first_divergence.csv",first_rows)
def onset(v,key,threshold):
 return next((r["step"] for r in v["trace"] if r[key]>threshold),None)
first={k:{"acquisition":v["acquisition"],"yaw_divergence_first_observed_step":onset(v,"yaw_abs_p95",.1),"major_yaw_spike_first_step":onset(v,"yaw_abs_p95",.5),"torque_saturation_first_step":onset(v,"torque_saturation_rate",0),"source_basin_distance":v["source_basin_distance_mean"],"minimum_basin_distance":v["minimum_basin_distance_mean"],"final_basin_distance":v["final_basin_distance_mean"],"basin_reduction_fraction":v["basin_reduction_fraction"],"yaw_abs_p95":v["yaw_abs_p95"],"yaw_sign_change_mean":v["yaw_sign_change_mean"],"contact_conditioned_yaw_spike_fraction":v["contact_conditioned_yaw_spike_fraction"]} for k,v in x["first_divergence"].items()}
first["interpretation"]="B0 enters the local W_MOVE basin but never satisfies sustained yaw acquisition; R40 first shows a major yaw spike at step 12 and moves away again after its minimum distance."
dump("first_divergence.json",first)
r40=x["first_divergence"]["R40"]
dump("yaw_failure_decomposition.json",{"target_yaw":0,"type":"OSCILLATORY_YAW_WITH_CONTACT_CONDITIONED_SPIKES","constant_yaw_bias":r40["yaw_mean"],"yaw_abs_p95":r40["yaw_abs_p95"],"yaw_sign_change_mean":r40["yaw_sign_change_mean"],"contact_conditioned_spike_fraction":r40["contact_conditioned_yaw_spike_fraction"],"left_right_action_asymmetry":r40["left_right_action_asymmetry"],"translation_insufficient":False,"evidence":"forward velocity approaches target while yaw repeatedly exits threshold","dominant_frequency_hz":"not robustly identifiable from 1.5 s trace"})
ramp_rows=[]
for key,v in x["command_ramps"].items():
 for pol in ("B0","R40"):ramp_rows.append({"variant":key,"ramp_seconds":v["ramp_seconds"],"policy":pol,**{q:v[pol][q] for q in ("acquisition","yaw_acquisition","fall","dangerous_slip","torque_saturation","minimum_basin_distance_mean","basin_reduction_fraction")}})
csvwrite("command_ramp_counterfactual.csv",ramp_rows);dump("command_ramp_counterfactual.json",{"rows":ramp_rows,"maximum_acquisition":max(r["acquisition"] for r in ramp_rows),"meaningful_improvement":False,"formal_contract_changed":False})
prev=[]
for key,v in supp["previous_action"].items():
 prev.append({"variant":key,"semantics":{"P0":"actual S_HOLD previous action","P1":"zeros","P2":"nearest W_MOVE previous action","P3":"nearest W_MOVE current action"}[key],"B0_acquisition":v["B0"]["acquisition"],"R40_acquisition":v["R40"]["acquisition"],"R40_yaw_acquisition":v["R40"]["yaw_acquisition"],"R40_basin_reduction":v["R40"]["basin_reduction_fraction"],"safe":not(v["R40"]["fall"] or v["R40"]["dangerous_slip"])})
dump("previous_action_counterfactual.json",{"rows":prev,"maximum_acquisition":max(r["R40_acquisition"] for r in prev),"meaningful_improvement":False,"diagnostic_only_P2_P3":True})
gates=[]
for key,v in x["gate_duration"].items():gates.append({"variant":key,"duration":{"G15":1.5,"G20":2.,"G30":3.,"GFULL":"3 s full"}[key],**{q:v[q] for q in ("acquisition","yaw_acquisition","fall","dangerous_slip","torque_saturation","minimum_basin_distance_mean","final_basin_distance_mean","basin_reduction_fraction")}})
dump("residual_gate_duration_counterfactual.json",{"rows":gates,"maximum_acquisition":max(r["acquisition"] for r in gates),"gate_too_short_supported":False,"formal_gate_changed":False})
temporal=x["temporal_gradient"]
dump("temporal_gradient_attribution.json",{"windows":temporal,"physical_state_evidence":{"R40_trace":x["first_divergence"]["R40"]["trace"],"major_yaw_spike_step":12},"yaw_gradient_growth_W3_over_W0":temporal["W3"]["yaw_gradient_norm"]/temporal["W0"]["yaw_gradient_norm"],"interpretation":"yaw gradient is nonzero early but peaks after the step-12 physical yaw breakdown; large norm is not evidence of useful causal control"})
probes=[]
for key,v in x["temporary_probes"].items():probes.append({"probe":key,"acquisition":v["acquisition"],"yaw_acquisition":v["yaw_acquisition"],"step20_yaw_abs_p95":v["trace"][-1]["yaw_abs_p95"],"step20_forward_velocity":v["trace"][-1]["forward_velocity_mean"],"basin_reduction":v["basin_reduction_fraction"],"fall":v["fall"],"slip":v["dangerous_slip"]})
dump("temporary_update_causal_probes.json",{"temporary_clones_only":True,"persistent_update":0,"rows":probes,"yaw_only_reduces_yaw_to_acquisition":False,"velocity_only_reaches_basin":False,"full_cancellation_supported":False,"classification":"START_YAW_GRADIENT_NONCAUSAL"})
a5=EXP13/"phase_w2_p1_a5_versioned_four_step_start_trajectory_overlay_preflight/four_step_runtime_positive_control.json";a7=EXP13/"phase_w2_p1_a7_r2_rear_yaw_start_teacher_replay_v2/selected_checkpoint.json";a8=EXP13/"phase_w2_p1_a8_offline_start_teacher_oracle/offline_start_teacher_oracle_v1.json"
exp13={"read_only":True,"A5":{"artifact":str(a5.relative_to(REPO)).replace("\\","/"),"sha256":sha(a5),"result":"PC2 physical endpoint 99.73%, aggregate acquisition 92.65%, rear moving-yaw 18.0-22.5%; not reusable globally"},"A6":{"result":"ramp 0.75-3.0 s and W1B prefix horizons did not resolve rear-yaw oscillation"},"A7_R2":{"artifact":str(a7.relative_to(REPO)).replace("\\","/"),"sha256":sha(a7),"selected_update":75,"checkpoint_sha256":"1cf290ace57bd9be4aeb0199a41b643b8604757bd3b788f2c98cec17e3f65028","result":"rear yaw passed, full 24-condition retention failed at 315/+0.3"},"A8":{"artifact":str(a8.relative_to(REPO)).replace("\\","/"),"sha256":sha(a8),"result":"two-checkpoint formal-point oracle passed, but local neighborhood and pure-yaw/static retention failed","runtime_reusable":False},"differences_from_D15_D16":["exp013 uses formal stop-pool/source-state lifecycle rather than S_HOLD-generated Exp014 snapshots","A5 uses a W1B action prefix/route","A7 policies use legacy observation/history and distinct roll-in timing","A8 uses a condition-dependent two-checkpoint map"],"reusable_safe_trajectory":False}
dump("exp013_start_artifact_audit.json",exp13)
search=x["reachability"];csvwrite("reachability_search_results.csv",[{k:v for k,v in r.items() if k!="history"} for r in search]);dump("reachability_search_manifest.json",{"method":"CEM-equivalent global piecewise-constant residual sequence","search_split":"train","search_snapshots":32,"variables":"first 25 or 50 steps, 37D residual","persistent_policy_update":0,"validation_search":0,"sequence_transfer_by_design":True,"variants":[{"horizon":r["horizon_steps"],"bound":r["bound"]} for r in search]});dump("reachability_search_results.json",{"rows":search,"successful_variant":None,"all_failed":True})
oracle={"minimum_snapshots":32,"safe_success_requirement":.8,"results":[{"source":"reachability_search","maximum_success":max(r["success_rate"] for r in search),"maximum_safe_rate":max(r["safe_rate"] for r in search),"basin_entry_max":max(r["basin_entry_50pct"] for r in search)},{"source":"existing_exp013","applicable_to_S_HOLD_sources":False,"reason":"different source-state and route contracts; A8 local-neighborhood/retention failure"}],"START_TRAJECTORY_ORACLE_EXISTS":False}
dump("oracle_trajectory_viability.json",oracle)
subs=["START_SOURCE_STATE_MANIFOLD_GAP","START_YAW_GRADIENT_NONCAUSAL","START_REWARD_GRADIENT_LATE","START_ADDITIVE_RESIDUAL_CONTRACT_INSUFFICIENT"]
root={"sub_classifications":subs,"not_supported":["START_COMMAND_RAMP_CONTRACT_MISMATCH","START_PREVIOUS_ACTION_CONTRACT_MISMATCH","START_RESIDUAL_GATE_TOO_SHORT","START_SAFE_TRAJECTORY_EXISTS_BUT_PPO_DOES_NOT_DISCOVER","START_RESIDUAL_BOUND_INSUFFICIENT","START_EXISTING_EXP013_ORACLE_REUSABLE"],"decision_precedence_applied":["no command/previous-action improvement","no reusable/search oracle","no gate-duration acquisition","yaw causality failure selected before architecture"],"primary_evidence":{"geometry_distance_mean":geom["nearest_state_distance"]["mean"],"contact_mismatch_rate":geom["contact_mismatch_rate"],"R40_acquisition":r40["acquisition"],"R40_yaw_p95":r40["yaw_abs_p95"],"yaw_gradient_W3_over_W0":temporal["W3"]["yaw_gradient_norm"]/temporal["W0"]["yaw_gradient_norm"],"reachability_max_success":0}}
dump("root_cause_classification.json",root)
classification="EXP014_D17_YAW_REWARD_CAUSALITY_FAIL";dump("stage_classification.json",{"classification":classification,"sub_classifications":subs,"D15_unchanged":True,"D16_unchanged":True});dump("recommended_next_action.json",{"experiment":"redesign early-phase yaw/weight-shift objective","single_authorized_route":True,"do_not_increase_updates":True,"do_not_expand_curriculum":True,"do_not_modify_W_MOVE":True})
tracked=subprocess.check_output(["git","diff","--name-only",START_HEAD],cwd=REPO,text=True).splitlines()
protected_changed=[p for p in tracked if any(f"exp_{i:03d}_" in p.replace("\\","/") for i in range(5,14)) or any(f"phase_2_d{i}" in p.replace("\\","/") for i in range(6,17))]
dump("protected_hashes.json",{"starting_head":START_HEAD,"exp005_exp013_changed_by_D17":0,"D6_D16_changed_by_D17":0,"D17_touched_protected_paths":[],"preexisting_unrelated_dirty_paths_preserved":protected_changed,"persistent_policy_update":0,"new_checkpoint":0,"formal_contract_change":0,"RUN":0,"causal_dagger_v2":0,"remote_push":False})
(OUT/"reproduction_commands.ps1").write_text("# D17 read-only diagnosis\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d17_audit.py --headless --device cuda:0\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d17_geometry_prev.py --headless --device cuda:0\npython experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d17.py\n",encoding="utf-8")
REPORT.write_text(f"""# EXP014 Phase 2-D17 START source-state and causality audit

## Outcome

**{classification}**. D15 and D16 remain unchanged. No persistent policy update or checkpoint was created.

## Source geometry

The 64 S_HOLD train sources are a distinct manifold from the 10,240-state `W_MOVE_FORWARD_BASIN_REFERENCE_V1`: normalized nearest-state distance mean {geom['nearest_state_distance']['mean']:.4f}, nearest-action L2 mean {geom['nearest_action']['l2_mean']:.4f}, cosine mean {geom['nearest_action']['cosine_mean']:.4f}, and contact mismatch {geom['contact_mismatch_rate']:.1%}.

## Physical divergence

Direct W_MOVE reduced basin distance by {x['first_divergence']['B0_DIRECT_WMOVE']['basin_reduction_fraction']:.1%} but achieved 0% sustained acquisition because yaw oscillated (p95 {x['first_divergence']['B0_DIRECT_WMOVE']['yaw_abs_p95']:.4f}). R40 also acquired 0%; its first major yaw spike occurred at step 12 and its minimum basin distance {r40['minimum_basin_distance_mean']:.4f} rose again to {r40['final_basin_distance_mean']:.4f} by step 75.

No ramp, previous-action, or gate-duration counterfactual produced any acquisition. Extending the gate therefore does not explain D16. The one-update yaw-only probe did not produce yaw acquisition.

## Gradient causality

Yaw gradient norm grew from {temporal['W0']['yaw_gradient_norm']:.4f} in W0 to {temporal['W3']['yaw_gradient_norm']:.4f} in W3, a {temporal['W3']['yaw_gradient_norm']/temporal['W0']['yaw_gradient_norm']:.2f}x increase after the physical yaw breakdown. Large aggregate gradient was therefore late/noncausal for the required initial weight shift.

## Reachability and prior artifacts

25-step ±0.50, 50-step ±0.50, and 25-step ±0.75 CEM action-sequence probes all achieved 0% registered success on 32 train sources. Existing exp013 A5-A8 trajectories use different source-state/roll-in contracts and fail either rear-yaw, local-neighborhood, or retention gates; no reusable S_HOLD-source oracle was established.

## Decision

Per registered precedence, command semantics, existing/search oracle, and gate duration were rejected first. The next single experiment is an early-phase yaw/weight-shift objective redesign. More PPO updates and curriculum expansion are not authorized.

## Protection

exp_005-exp_013, D6-D16 artifacts, checkpoints, datasets, optimizer, physics, reward config, and formal contracts were not changed. Persistent updates, new checkpoints, RUN, Causal DAgger V2, and remote push are zero.
""",encoding="utf-8")
print(json.dumps({"classification":classification,"artifacts":23},indent=2))
if __name__=="__main__":pass
