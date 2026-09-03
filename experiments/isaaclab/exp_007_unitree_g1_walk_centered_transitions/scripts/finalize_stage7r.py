"""Finalize Stage 7R R0 interface result."""
import csv,hashlib,json,subprocess
from pathlib import Path
H=Path(__file__).resolve();EXP=H.parent.parent;REPO=EXP.parents[2];O=REPO/"results/exp_007_unitree_g1_walk_centered_transitions/stage7r_walk_to_run_152d";O.mkdir(parents=True,exist_ok=True)
S7=REPO/"results/exp_007_unitree_g1_walk_centered_transitions/stage7_walk_to_run"
def j(n,x):(O/n).write_text(json.dumps(x,indent=2)+"\n",encoding="utf-8")
layout=[(0,6,"current_skill_one_hot"),(6,12,"previous_skill_one_hot"),(12,13,"sin_target_heading_error"),(13,14,"cos_target_heading_error"),(14,16,"skill_local_target_state"),(16,17,"relative_target_pelvis_height"),(17,21,"skill_local_auxiliary_state"),(21,22,"target_vertical_velocity"),(22,23,"normalized_elapsed_time"),(23,24,"normalized_remaining_time"),(24,25,"skill_phase"),(25,26,"transition_progress"),(26,27,"recovery_mode"),(27,29,"target_posture_roll_pitch")]
j("stage7_reference.json",{"status":"FAIL","path":str(S7.relative_to(REPO)),"direct_full_edge":.586,"preserved":True})
j("interface_audit.json",{"actor_class":"G1CommandResidualActor","legacy_dim":123,"command_dim":29,"total_dim":152,"action_dim":37,"action_scale":.5,"composition":"running_base_action + 0.25*tanh(RUN residual)","RUN_route":0,"source":"exp_006 command_observation.py/residual_actor.py"})
j("command_observation_layout.json",{"absolute_indices":[{"start":123+a,"end":123+b,"name":n} for a,b,n in layout],"legacy_command":[9,12],"RUN_one_hot_index":123,"previous_RUN_index":129,"transition_progress_index":148,"reserved_field":None})
j("actor_architecture.json",{"base":"123->[256,128,128]->37 frozen","run_command_encoder":"29->[64]->32 trainable","run_state_adapter":"123->[128]->64 trainable","run_residual_head":"96->[64,64]->37 trainable","residual_scale":.25,"output":"full 37D base-plus-residual action"})
j("transition_actor_design_decision.json",{"route":"A","reason":"Existing semantics already contain RUN identity, heading, local target, elapsed/remaining, phase and transition progress; no field is redefined.","initialization":"strict deep copy of RUN_LOW actor","separate_checkpoint_required":True})
j("trainable_action_term.json",{"class":"WalkToRunTransitionAction","config":"WalkToRunTransitionActionCfg","observation":152,"action":37,"scale":.5,"runtime_blend":False,"global_previous_action":True})
with (S7/"direct_switch_baseline_episodes.csv").open(newline="",encoding="utf-8") as f:r=list(csv.DictReader(f))
valid=[x for x in r if x["source_contract"]=="True"]
counts={}
for x in valid:counts[x["contact_phase"]]=counts.get(x["contact_phase"],0)+1
j("source_occupancy_distribution.json",{"source":"actual frozen STAND->STAND_TO_WALK->WALK Stage7 diagnostic","valid":len(valid),"attempts":len(r),"note":"available for audit, not yet connected to PPO rollout buffer"})
j("source_phase_counts.json",counts)
j("transition_observation_contract.json",{"legacy_123":["base_linear_velocity","base_angular_velocity","projected_gravity","legacy_velocity_command","joint_position","joint_velocity","global_previous_action"],"command_29":"command_observation_layout.json","world_xy":False})
j("completion_detector.json",{"source":"Stage6 RUN_LOW","speed_ratio":.75,"speed_error":.2,"heading":.12,"flight_events":4,"safe_cycles":3,"alternation":.8,"valid_landing":.8,"flight_duration":[.04,.16]})
j("reward_definition.json",{"status":"FROZEN_NOT_EXECUTED","precursor":["safe_liftoff","safe_flight","valid_landing","alternating_landing","consecutive_cycles"],"safety":["fall","torso_contact","slip","impact","flight","ankle_dwell","knee_dwell","joint_limit","action_rate"]})
j("command_profile.json",{"source":1.2,"targets":[2.4,2.6,2.8],"candidates":[1.,1.4,1.8],"selected":None,"stage7_1p4_result":.259})
j("negative_control.json",{"transition_term_disabled":"Stage7 direct-switch reproduced","full_edge":.586,"model_0_action":"static finite forward verified","environment_equivalence":"PPO occupancy-buffer equivalence not established"})
j("pilot_results.json",{"executed":False,"reason":"R0 gate FAIL"})
j("checkpoint_sweep.json",{"executed":False});j("formal_summary.json",{"status":"NOT_RUN"});j("per_seed_results.json",{});j("per_target_results.json",{})
for n in ("episodes.csv","transition_timelines.csv","cycle_acquisition_metrics.csv"):(O/n).write_text("status\nNOT_RUN\n",encoding="utf-8")
j("action_discontinuity.json",{"status":"NOT_RUN"});j("impact_diagnostics.json",{"status":"NOT_RUN"});j("saturation_diagnostics.json",{"status":"NOT_RUN"});j("failure_counts.json",{"transition_action_interface_failure":1})
protected={"WALK":"9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa","RUN":"60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266","STAND":"734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"}
j("routing_preflight.json",{"runtime_blend":False,"production_controller_count":1,"RUN_TO_WALK_loaded":False,"protected_hashes":protected})
r0=json.loads((O/"r0_interface_gate.json").read_text())
j("gate.json",{"stage":"7R","status":"FAIL","eligible_for_stage8":False,"r0":r0,"failures":["actual WALK occupancy is not integrated into PPO rollout/advantage buffer","transition-only PPO segment isolation is not implemented"],"pilot_executed":False,"formal_executed":False,"artifact_created":False,"git_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()})
(O/"reproduction_commands.ps1").write_text('cd "$HOME\\workspace\\physical-ai-lab"\n& "$env:USERPROFILE\\workspace\\IsaacLab\\isaaclab.bat" -p experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\audit_stage7r_interface.py\n',encoding="utf-8")
print(json.dumps(json.loads((O/"gate.json").read_text()),indent=2))
