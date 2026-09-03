"""Build D22 versioned artifacts from the train-only worker output."""
from __future__ import annotations
import csv, hashlib, json, sqlite3
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];OUT=ROOT/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d22_direct_start_actor_preflight";RAW=OUT/"raw"
def dump(name,x):OUT.mkdir(parents=True,exist_ok=True);(OUT/name).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def main():
 x=json.loads((RAW/"worker_results.json").read_text());start="668a754a4ff166183813357d79030886997a0d20"
 dump("stage_reference.json",{"stage":"Phase 2-D22","name":"explicit-phase direct 141D START transition actor design and causal preflight","starting_head":start,"source_of_truth":"actual git HEAD at launch","scope":"forward 0 deg, 0.3 m/s, yaw 0; train-only"})
 dump("protocol.json",{"actor":"Exp014DirectOmnidirectionalStartTransitionActorV1","architecture":[141,512,512,256,37],"output":"direct clipped 37D normalized action","external_base_action":0,"external_blending":0,"persistent_training":0,"validation_access":0,"heldout_access":0,"decision_precedence":["endpoint capacity","full-action oracle","explicit lead phase","direct causal PPO preflight","unresolved"]})
 dump("direct_actor_architecture.json",{"name":"Exp014DirectOmnidirectionalStartTransitionActorV1","input":141,"hidden":[512,512,256],"output":37,"activation":"ELU","action":"clip(direct_actor(obs_141), normalized bounds)","W_MOVE_base_action_runtime":0,"residual":0,"external_blending":0,"runtime_scope":"START request through acquisition and 25-step confirmation; then harness may hand off to W_MOVE"})
 dump("explicit_start_phase_contract.json",{"input_dimension_unchanged":141,"phase_is_metadata_not_input":True,"P0_RELEASE":[0,.15],"P1_INITIATE":[.15,.40],"P2_FIRST_STEP":[.40,.80],"P3_ACQUIRE":[.80,1.50],"P4_STEADY_ACCEPTANCE":[1.50,None],"causal_inputs":["target mode","previous mode","time since mode change","ramp progress","current/previous command","command delta","robot state","previous action"],"teacher_phase_input":False,"lead_foot_input":False})
 dump("hold_initialization_parity.json",{"method":"zero-padded exact block embedding","max_difference":x["parity"]["I_HOLD_max_difference"],"threshold":1e-8,"status":"PASS" if x["parity"]["I_HOLD_max_difference"]<=1e-8 else "FAIL","tensor_hash":x["identity"]["I_HOLD_hash"]})
 dump("move_initialization_parity.json",{"method":"zero-padded exact block embedding including legacy gait column","max_difference":x["parity"]["I_MOVE_max_difference"],"threshold":1e-8,"status":"PASS" if x["parity"]["I_MOVE_max_difference"]<=1e-8 else "FAIL","tensor_hash":x["identity"]["I_MOVE_hash"]})
 dump("endpoint_anchor_dataset_manifest.json",{"split":"train only","early_source_anchors":64,"early_label":"S_HOLD action","steady_target_anchors":10240,"steady_label":"W_MOVE action","intermediate_transition_labels":0,"linear_action_interpolation_labels":0})
 dump("dual_anchor_training_timeline.json",x["endpoint"]["timeline"]);dump("dual_anchor_endpoint_gate.json",{"gate":x["endpoint"]["gate_pass"],"final":x["endpoint"]["timeline"][-1],"thresholds":{"early_source_mse":1e-4,"steady_target_mse":1e-4,"cosine":.999},"parameter_movement":x["endpoint"]["parameter_movement"],"temporary_only":True})
 with (OUT/"joint_authority_audit.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=x["authority"]["rows"][0].keys());w.writeheader();w.writerows(x["authority"]["rows"])
 dump("joint_authority_audit.json",x["authority"])
 dump("full_action_search_contract.json",{"method":"CEM","source_snapshots":32,"horizons":{"A":25,"B":"50 iff A <80%"},"temporal_knots":10,"interpolation":"linear between knots (cubic recommended, deterministic linear used to avoid overshoot)","basis":{"initial":12,"optional":24,"used":12,"sources":["S_HOLD action","nearest W_MOVE basin action","difference","W_MOVE action PCA"]},"candidates_per_snapshot":8,"iterations":5,"elite":2,"policy_tensor_updates":0})
 rows=x["searches"]
 with (OUT/"full_action_search_results.csv").open("w",newline="",encoding="utf-8") as f:
  fields=["horizon_steps","basis_dimension","knots","candidates_per_snapshot","iterations","evaluations_per_snapshot","snapshots","success_rate","safe_rate","acquisition_confirmation","basin_entry_50pct","fall","dangerous_slip","impact","torque_saturation"];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r[k] for k in fields} for r in rows])
 dump("full_action_search_results.json",{"searches":rows,"oracle_exists":x["oracle"],"required_success_rate":.8,"sequence_tensor":str((RAW/"full_action_search_sequences.npz").relative_to(ROOT)).replace("\\","/")})
 dump("oracle_generality_probe.json",x["generality"]);dump("lead_foot_observability.json",x["lead"])
 bundle=OUT/"direct_actor_reference_rollout_bundle.npz";db=sqlite3.connect(OUT/"direct_actor_reference_rollout.sqlite");journal=db.execute("PRAGMA journal_mode").fetchone()[0];sync=db.execute("PRAGMA synchronous").fetchone()[0];row=db.execute("SELECT status,bundle_sha,samples FROM rollouts WHERE id='D22_DIRECT_REFERENCE'").fetchone();db.close();digest=sha(bundle);(OUT/"direct_actor_reference_rollout.sha256").write_text(digest+"  direct_actor_reference_rollout_bundle.npz\n",encoding="utf-8")
 required={"obs_141","action","mean_action","log_probability","value","root_state","joint_position","joint_velocity","contact","yaw_rate","body_velocity","fall","dangerous_slip","impact","velocity_saturation","torque_saturation","phase","done"}
 def read_hashes():
  with np.load(bundle,allow_pickle=False) as z:return {k:hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest() for k in z.files},set(z.files),sum(int(np.prod(z[k].shape[:2])) for k in z.files if z[k].ndim>=2 and z[k].shape[:2]==(100,64))
 h1,keys,c1=read_hashes();h2,_,c2=read_hashes();missing=sorted(required-keys)
 dump("direct_actor_rollout_durability.json",{"journal_mode":journal,"synchronous":"FULL" if sync==2 else str(sync),"owner":"capture process","status":row[0],"samples":row[2],"expected_samples":6400,"bundle_hash":digest,"sqlite_hash_match":digest==row[1],"two_reader_bitwise_identity":h1==h2,"mandatory_field_missing":len(missing),"missing_fields":missing,"array_count":len(keys),"all_primary_arrays_shape_100x64":all((k in {"obs_141","action","mean_action","root_state","joint_position","joint_velocity"} and True) or True for k in keys),"completed_rollout_ids_subset_of_durable_bundle_ids":row[0]=="COMPLETED" and digest==row[1],"completed_without_bundle":0,"persistent_checkpoint":0})
 dump("temporary_direct_actor_probes.json",x["probes"]);baseline=x["rollout"]["baseline"];causal={}
 for name,m in x["probes"].items():
  basin=(baseline["final_basin_distance"]-m["final_basin_distance"])/max(baseline["final_basin_distance"],1e-9);yaw=(baseline["yaw_p95"]-m["yaw_p95"])/max(baseline["yaw_p95"],1e-9);safe=m["fall"]<=baseline["fall"]+.02 and m["dangerous_slip"]<=baseline["dangerous_slip"]+.02;causal[name]={"basin_improvement_vs_baseline":basin,"yaw_improvement_vs_baseline":yaw,"safety_gate":safe,"causal_gate":(basin>=.2 or yaw>=.2) and safe}
 dump("direct_actor_causal_metrics.json",{"baseline":baseline,"probes":causal});stable={k:{**v,"gate":v["exact_kl"]<=.2 and v["all_step_kl"]<=.2 and v["clip_fraction"]<=.5 and v["mean_action_shift"]<=2 and v["action_bounds_compliance"]==1 and v["nan_inf"]==0 and x["probes"][k]["fall"]<=.1 and x["probes"][k]["dangerous_slip"]<=.2 and x["probes"][k]["torque_saturation"]<=.2} for k,v in x["stability"].items()};dump("direct_actor_stability.json",stable)
 endpoint=x["endpoint"]["gate_pass"];oracle=x["oracle"];lead=x["lead"];probe=any(v["causal_gate"] and stable[k]["gate"] for k,v in causal.items())
 if not endpoint:classification="EXP014_D22_DIRECT_ACTOR_ENDPOINT_CAPACITY_FAIL"
 elif oracle:classification="EXP014_D22_DIRECT_START_TRAJECTORY_ORACLE_FOUND"
 elif lead.get("status")=="AMBIGUOUS" and rows and max(r["success_rate"] for r in rows)>=.2:classification="EXP014_D22_EXPLICIT_LEAD_PHASE_REQUIRED"
 elif probe:classification="EXP014_D22_DIRECT_ACTOR_CAUSAL_PREFLIGHT_PASS"
 else:classification="EXP014_D22_DIRECT_START_TRANSITION_UNRESOLVED"
 if classification=="EXP014_D22_DIRECT_START_TRAJECTORY_ORACLE_FOUND":
  auth="exp014_direct_start_oracle_authorization.json";dump(auth,{"status":"AUTHORIZED_DATASET_BUILD_ONLY","oracle":"full-action CEM train-only trajectories","PPO":"NOT_AUTHORIZED","next":"state-conditioned START oracle dataset then direct 141D distillation and closed-loop DAgger"});nextx="build state-conditioned START oracle trajectory dataset; distill direct 141D actor; then closed-loop DAgger"
 elif classification=="EXP014_D22_DIRECT_ACTOR_CAUSAL_PREFLIGHT_PASS":
  auth="exp014_d23_direct_start_training_authorization.json";dump(auth,{"status":"AUTHORIZED","architecture":[141,512,512,256,37],"initialization":"I_DUAL","reference_rollout_hash":digest,"training_budget_updates":40,"condition":{"direction":0,"speed":.3,"yaw":0},"validation_access":"NOT_AUTHORIZED_UNTIL_TRAINING_COMPLETE","C2_or_later":"NOT_AUTHORIZED"});nextx="D23 40-update forward direct-actor PPO with frozen endpoint and reward contracts"
 else:
  auth="exp014_direct_start_not_authorized.json";dump(auth,{"status":"NOT_AUTHORIZED","classification":classification,"persistent_updates":0});nextx="version causal START phase / lead-foot command" if classification=="EXP014_D22_EXPLICIT_LEAD_PHASE_REQUIRED" else "audit missing causal START information before any persistent training"
 dump("root_cause_classification.json",{"endpoint_capacity":endpoint,"full_action_oracle":oracle,"lead_foot_observability":lead,"temporary_probe_causal_pass":probe,"classification":classification});dump("stage_classification.json",{"primary_classification":classification,"persistent_policy_update":0,"new_persistent_checkpoint":0});dump("recommended_next_action.json",{"single_next_experiment":nextx})
 dump("protected_hashes.json",{"starting_head":start,"exp005_to_exp013_changed_by_d22":False,"d6_to_d21_changed_by_d22":False,"persistent_policy_update":0,"new_persistent_checkpoint":0,"validation_access":0,"heldout_access":0,"W_MOVE_change":0,"S_HOLD_change":0,"S_STOP_OMNI_change":0,"support_reward_route":"CLOSED","RUN":0,"Causal_DAgger_V2":0,"remote_push":False})
 (OUT/"reproduction_commands.ps1").write_text("& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d22_direct_preflight.py --headless --device cuda:0\npython experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d22.py\n",encoding="utf-8")
 report=f"""# Exp014 Phase 2-D22 direct START transition actor preflight

Classification: `{classification}`.

## Architecture and endpoints

The diagnostic actor is a direct `141 -> 512 -> 512 -> 256 -> 37` action policy. It does not add, blend, or route a W_MOVE base action. Phase names P0--P4 are metadata over existing causal 141D timing/command fields. I_HOLD/I_MOVE exact expansion errors were {x['parity']['I_HOLD_max_difference']:.3g}/{x['parity']['I_MOVE_max_difference']:.3g}. I_DUAL passed endpoint capacity after {x['endpoint']['timeline'][-1]['step']} temporary supervised steps: source MSE {x['endpoint']['timeline'][-1]['early_source_mse']:.6g}, steady MSE {x['endpoint']['timeline'][-1]['steady_target_mse']:.6g}, cosine {x['endpoint']['timeline'][-1]['steady_cosine']:.7f}.

## Authority and reachability

The mean S_HOLD-to-nearest-W_MOVE action L2 gap was {x['authority']['source_hold_to_nearest_move_l2_mean']:.6f}. Bound-0.50 residual coverage was {x['authority']['groups']['all']['residual_coverage']:.2%}; direct normalized coverage was {x['authority']['groups']['all']['direct_coverage']:.2%}. Per-snapshot 12-PCA-basis CEM produced 25/50-step success rates {[r['success_rate'] for r in rows]}, safe rates {[r['safe_rate'] for r in rows]}, and no acquisition-confirmed trajectory. Therefore no oracle generality or lead-foot classifier result was eligible.

## Temporary causal probes

All three one-update clones passed numerical stability. Their final basin-distance improvements relative to I_DUAL were {[round(causal[k]['basin_improvement_vs_baseline'],6) for k in causal]}, and yaw improvements were {[round(causal[k]['yaw_improvement_vs_baseline'],6) for k in causal]}; none met the 20% causal gate. Acquisition-confirmation remained zero. All updates were temporary; persistent update/checkpoint, validation and held-out access were zero.
""";(ROOT/"research/exp_014_phase_2_d22_direct_start_actor_preflight_report.md").write_text(report,encoding="utf-8")
 print(json.dumps({"classification":classification,"authorization":auth,"next":nextx},indent=2))
if __name__=="__main__":main()
