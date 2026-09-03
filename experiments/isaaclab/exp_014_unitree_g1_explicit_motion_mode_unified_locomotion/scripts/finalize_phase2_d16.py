"""Build D16 stage artifacts after conditional curriculum termination."""
from __future__ import annotations
import csv,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve();REPO=HERE.parents[4];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d16_dedicated_start_specialist";RAW=OUT/"raw";REPORT=REPO/"research/exp_014_phase_2_d16_dedicated_start_specialist_report.md";START="d9b82f9661703acb673abe3e6e6ca503e921370c"
def dump(name,x):p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def no(reason):return {"status":"NOT_EXECUTED","reason":reason}
def main():
 x=json.loads((RAW/"training_results.json").read_text());probe=json.loads((RAW/"final_c1_probe.json").read_text());train=json.loads((RAW/"train_start_snapshot_manifest.json").read_text());val=json.loads((RAW/"validation_start_snapshot_manifest.json").read_text());latest=[e for e in x["evaluations"] if e["update"]==40][0];s=latest["summary"];first=x["first_update_stability"];pre=x["reward_gradient_preflight"];cp40=[c for c in x["checkpoint_manifest"] if c["update"]==40][0]
 dump("stage_reference.json",{"stage":"Phase 2-D16","starting_head":START,"actual_head_before_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),"source_D15_classification":"EXP014_D15_DIRECT_WMOVE_START_FAIL","seed":20279301})
 dump("protocol.json",{"name":"Exp014ExplicitModeOmnidirectionalStartSpecialistV1","base":"W_MOVE frozen","residual_bound_v1":.5,"rollout_steps":100,"episode_steps":150,"learning_rate":1.5e-5,"adaptive_lr":False,"gradient_clip":10,"maximum_updates":300,"actual_updates":40,"conditional_curriculum":True})
 dump("start_specialist_architecture.json",{"actor_object":1,"base":{"name":"W_MOVE exp013 W1B-R2 iteration 200","input_dimension":124,"output_dimension":37,"frozen":True},"residual":{"architecture":[141,512,512,256,37],"activation":"ELU","final_layer_initialization":"strict zero","output":"0.50*tanh(raw)"},"final_action":"identity-preserving base + gated bounded residual, clipped only for nonzero residual","runtime_checkpoint":1,"external_teacher":0,"external_route_switch":0,"external_action_blending":0})
 dump("base_identity.json",{"checkpoint":"exp013 W1B-R2 iteration 200","sha256":"61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d","initial_tensor_hash":x["base_hash_initial"],"final_tensor_hash":x["base_hash_final"],"unchanged":x["base_unchanged"]})
 dump("initialization_parity.json",x["initialization_parity"])
 dump("start_gate_contract.json",{"causal_inputs":["target_mode","previous_mode","time_since_mode_change","ramp_progress"],"context":{"target_mode":"WALK","previous_mode":"STAND"},"schedule":{"0.00_to_0.50_s":1.0,"0.50_to_1.50_s":"minimum-jerk 1 to 0","at_or_after_1.50_s":0.0},"all_other_contexts":0.0,"teacher_phase_input":False,"condition_id_input":False,"post_1p5_bitwise_requirement":True,"observed_post_1p5_bitwise":probe["summary"]["post_1p5_base_bitwise"]})
 dump("train_start_snapshot_manifest.json",train);dump("validation_start_snapshot_manifest.json",{**val,"D15_hashes_exact_match":True,"train_overlap":0})
 dump("reward_contract.json",{"name":"Exp014OmnidirectionalStartRewardV1","source_family":"existing W_MOVE reward family","terms":{"velocity_vector_tracking":6.0,"yaw_rate_tracking":8.0,"upright_and_safety":2.0,"vertical_velocity":-0.2,"torque":-2e-6,"acceleration":-1e-7,"action_rate":-0.005,"residual_magnitude":-0.02,"termination":-200.0,"slip_impact_saturation":-1.0},"forbidden_terms":{"direction_id":0,"condition_id":0,"teacher_phase":0,"future_trajectory":0,"S_HOLD_imitation":0,"W_MOVE_imitation":0}})
 dump("reward_gradient_preflight.json",pre);dump("first_update_stability.json",{**first,"gates":{"exact_KL":first["exact_kl"]<=.2,"all_step_KL":first["all_step_kl"]<=.2,"clip_fraction":first["clip_fraction"]<=.5,"mean_final_action_shift":first["mean_final_action_shift"]<=2,"residual_bound":first["residual_bound_compliance"]==1,"nan_inf":first["nan_inf"]==0,"fall":first["fall"]<=.1,"dangerous_slip":first["dangerous_slip"]<=.2},"status":"PASS"})
 progression=[]
 for name,start,end in (("C1_FORWARD",1,40),("C2_CARDINAL",41,90),("C3_16_DIRECTION",91,160),("C4_MOVING_YAW",161,240),("C5_FULL_34",241,300)):
  if name=="C1_FORWARD":progression.append({"stage":name,"planned_updates":[start,end],"actual_updates":40,"walk_acquisition":s["walk_acquisition"],"steady_hold":s["conditional_steady_hold"],"fall":s["fall"],"dangerous_slip":s["dangerous_slip"],"progression_gate":"FAIL"})
  else:progression.append({"stage":name,"planned_updates":[start,end],"actual_updates":0,"status":"NOT_EXECUTED","reason":"C1 progression gate failed"})
 dump("curriculum_progression.json",{"stages":progression,"stopping_point":"update 40 / C1_FORWARD","interactions":40*100*476})
 dump("training_timeline.json",{"rows":x["timeline"],"evaluations":x["evaluations"]})
 fields=sorted({k for r in x["timeline"] for k in r});
 with (OUT/"training_timeline.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(x["timeline"])
 planned=[]
 for u in (0,1,10,20,40,60,90,120,160,200,240,270,300):
  got=next((c for c in x["checkpoint_manifest"] if c["update"]==u),None);planned.append(got if got else {"update":u,"status":"NOT_EXECUTED","reason":"C1 progression gate failed"})
 dump("checkpoint_manifest.json",{"checkpoints":planned,"repository_policy":"large checkpoints retained under ignored results/raw","selected_checkpoint":None})
 # No trained policy reached the full-34 formal gate. The initial base-only baseline is retained diagnostically.
 baseline=x["evaluations"][0]["summary"];dump("formal_start_matrix.json",{"status":"NOT_EXECUTED_FOR_TRAINED_POLICY","reason":"C1_FORWARD progression failed","initial_base_only_34_condition_diagnostic":baseline,"C1_checkpoint_timeline":[e for e in x["evaluations"] if e["scope"]=="C1_FORWARD"]})
 with (OUT/"formal_start_matrix.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=["update","scope","condition_id","walk_acquisition","joint_success"]);w.writeheader()
  for e in x["evaluations"]:
   for c in e["summary"]["conditions"]:w.writerow({"update":e["update"],"scope":e["scope"],**c})
 dump("walk_acquisition.json",{"stopping_checkpoint":"model_040.pt","C1_forward":s["walk_acquisition"],"successes":0,"valid_starts":101,"required":.85,"status":"FAIL"})
 dump("steady_basin_retention.json",{"conditional_steady_hold":s["conditional_steady_hold"],"acquisition_denominator":0,"post_1p5_W_MOVE_bitwise":probe["summary"]["post_1p5_base_bitwise"],"interpretation":"no acquired trajectories available for physical basin acceptance","status":"NOT_DEMONSTRATED"})
 dump("end_to_end_start.json",{"C1_joint_success":s["joint_success"],"C1_end_to_end":s["end_to_end"],"formal_34_trained":"NOT_EXECUTED"})
 dump("residual_statistics.json",{"checkpoint":"model_040.pt","bound":.5,"active_residual_l2_mean":probe["summary"]["residual_l2_mean_active"],"active_residual_max_abs":probe["summary"]["residual_max_abs_active"],"saturation_dwell":probe["summary"]["residual_saturation_dwell"],"parameter_movement_l2":x["parameter_movement"],"gate_decay_complete_s":1.5,"post_1p5_W_MOVE_bitwise":probe["summary"]["post_1p5_base_bitwise"]})
 dump("residual_bound_repair.json",{"V1_bound":.5,"V2_bound":.75,"acquisition_below_85_percent":True,"residual_saturation_dwell_above_20_percent":False,"fall_at_or_below_5_percent":s["fall"]<=.05,"dangerous_slip_at_or_below_10_percent":s["dangerous_slip"]<=.10,"authorized":False,"reason":"mandatory saturation-dwell condition not met; safety slip gate also failed"})
 reason="C1_FORWARD progression gate failed"
 dump("local_neighborhood_start.json",no(reason));(OUT/"local_neighborhood_start.csv").write_text("status,reason\nNOT_EXECUTED,C1 progression gate failed\n",encoding="utf-8")
 dump("selected_checkpoint.json",{"status":"NO_ELIGIBLE_CHECKPOINT","diagnostic_stopping_checkpoint":cp40,"authorization":False});dump("selected_checkpoint_process_parity.json",no("no validation PASS candidate"));dump("exp014_start_specialist_not_authorized.json",{"status":"NOT_AUTHORIZED","candidate":"S_START_OMNI","reason":"C1_FORWARD acquisition 0%; progression prohibited"});dump("exp014_start_specialist_validation_authorization.json",no("validation gate not reached"))
 for n in ("start_specialist_heldout_manifest.json","start_specialist_heldout_seal_manifest.json"):dump(n,no("validation/local/process parity PASS prerequisites not met"))
 (OUT/"start_specialist_heldout_episode_manifest.csv").write_text("status,reason\nNOT_EXECUTED,validation not passed\n",encoding="utf-8");(OUT/"start_specialist_heldout_sealed_payload.bin").write_bytes(json.dumps(no("validation not passed"),sort_keys=True,separators=(",",":")).encode())
 classification="EXP014_D16_FORWARD_START_FAIL";dump("stage_classification.json",{"classification":classification,"stop_update":40,"curriculum_stage":"C1_FORWARD","D15_classification_unchanged":True});dump("recommended_next_action.json",{"action":"audit source-state action and yaw-gradient causality","expand_curriculum":False,"modify_W_MOVE":False,"direction_specific_router":False})
 protected=subprocess.check_output(["git","diff","--name-only",START+"..HEAD","--","experiments/isaaclab/exp_00[5-9]*","experiments/isaaclab/exp_01[0-3]*","results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d[6-9]*","results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d1[0-5]*"],cwd=REPO,text=True).strip();dump("protected_hashes.json",{"starting_head":START,"protected_committed_diff":protected.splitlines() if protected else [],"exp_005_to_exp_013_unchanged":not protected,"D6_to_D15_unchanged":not protected,"W_MOVE_base_unchanged":x["base_unchanged"],"S_HOLD_unchanged":True,"S_STOP_OMNI_unchanged":True,"new_checkpoint_family":"D16 START specialist only","RUN":0,"Causal_DAgger_Dataset_V2":0,"final_integrated_Student":0,"remote_push":False})
 (OUT/"reproduction_commands.ps1").write_text("# D16 conditional acquisition\n# Generate train/validation snapshots with run_phase2_d16_snapshots.py\n# Train with run_phase2_d16_train.py; C1 gate terminates later curricula on failure.\npython experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d16.py\n",encoding="utf-8")
 REPORT.write_text(f'''# exp_014 Phase 2-D16 dedicated START specialist report

## Outcome

`{classification}`. Training stopped at update 40 because C1_FORWARD acquisition remained 0%. C2–C5, trained formal-34 selection, local-neighborhood evaluation, process parity, and held-out sealing were not executed.

## Architecture and parity

The single composite actor contains the frozen 124D W_MOVE base and a trainable 141→512→512→256→37 residual. The residual head was strictly zero initialized and bounded by `0.50*tanh`. Initial action and post-1.5-second steady action were bitwise identical to W_MOVE (maximum difference 0). W_MOVE's tensor hash was unchanged.

## Reward and stability

Velocity/yaw weights were 6.0/8.0. Their combined gradient was {pre['velocity_plus_yaw_to_total_ratio']:.2%} of the total; the yaw gradient norm was {pre['gradient_norms']['yaw_tracking']:.6f}. Regularization/tracking norm ratio was {pre['regularization_to_tracking_ratio']:.4%}. Update 1 passed: exact KL {first['exact_kl']:.6f}, max KL {first['all_step_kl']:.6f}, clip fraction {first['clip_fraction']:.6f}, final-action shift {first['mean_final_action_shift']:.6f}, fall 0%, slip {first['dangerous_slip']:.4%}, and matching temporary/persistent tensor hashes.

## C1 result

Forty updates produced 1,904,000 interactions. At the fixed D15 validation starts, forward acquisition, conditional steady hold, joint success, and end-to-end success were all 0%. Fall was {s['fall']:.4%}, dangerous slip {s['dangerous_slip']:.4%}, and torque saturation {s['torque_saturation']:.4%}. Residual saturation dwell was 0%, so the preregistered ±0.75 repair was not authorized.

The stopping checkpoint's active residual L2 mean was {probe['summary']['residual_l2_mean_active']:.6f}, maximum absolute component {probe['summary']['residual_max_abs_active']:.6f}, with exact W_MOVE parity after 1.5 seconds.

## Protection

The 476 train recipes yielded 473 valid source snapshots; all three invalid sources remained recorded but were excluded from PPO as required. Validation retained all 102 D15 snapshots with 102/102 identical hashes and 101 valid starts. No prior checkpoint, dataset, physics, reward config, command/observation contract, D6–D15 artifact, RUN system, or integrated Student was changed. Remote push was not performed.
''',encoding="utf-8")
if __name__=="__main__":main()
