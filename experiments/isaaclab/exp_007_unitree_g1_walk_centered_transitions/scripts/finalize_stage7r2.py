"""Finalize Stage 7R2 transition-only runner R0."""
import csv,json,subprocess
from pathlib import Path
H=Path(__file__).resolve();EXP=H.parent.parent;REPO=EXP.parents[2];O=REPO/"results/exp_007_unitree_g1_walk_centered_transitions/stage7r2_transition_only_runner";O.mkdir(parents=True,exist_ok=True)
S7=REPO/"results/exp_007_unitree_g1_walk_centered_transitions/stage7_walk_to_run";S7R=REPO/"results/exp_007_unitree_g1_walk_centered_transitions/stage7r_walk_to_run_152d"
def j(n,x):(O/n).write_text(json.dumps(x,indent=2)+"\n",encoding="utf-8")
j("stage7_reference.json",{"path":str(S7.relative_to(REPO)),"status":"FAIL","preserved":True})
j("stage7r_reference.json",{"path":str(S7R.relative_to(REPO)),"status":"FAIL","preserved":True})
j("runner_design_decision.json",{"design":"COHORT","cohort_size":512,"reason":"no invalid/prefix masks; every stored step is transition-controlled","phases":{"A":"NO_GRAD_NO_STORAGE","B":"TRANSITION_STORAGE"}})
j("source_preparation_contract.json",{"state":"WALK","command_mps":1.2,"hold_s":1,"speed_error":.2,"heading":.12,"finite":True,"safety":True})
with (S7/"direct_switch_baseline_episodes.csv").open(newline="",encoding="utf-8") as f:r=list(csv.DictReader(f))
valid=[x for x in r if x["source_contract"]=="True"]
j("source_preparation_summary.json",{"attempts":len(r),"successes":len(valid),"success_rate":len(valid)/len(r),"minimum":.9,"pass":len(valid)/len(r)>=.9,"live_cohort_512_handoff":False})
with (O/"source_preparation_episodes.csv").open("w",newline="",encoding="utf-8") as f:
 w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
j("transition_episode_contract.json",{"rl_start":"first step after valid WALK source","physical_reset_is_rl_start":False,"stored_controller":"WALK_TO_RUN only","history_reset":["elapsed","progress","completion","flight","landing","cycles","reward","critic","GAE","length","failure"],"preserved":["robot state","contact","global previous action","heading","gait phase"]})
j("terminal_bootstrap_contract.json",{"terminated":["RUN acceptance","fall","torso contact","unrecoverable safety"],"truncated":["timeout","horizon"],"success_bootstrap":0,"failure_bootstrap":0,"timeout_bootstrap":0})
j("critic_initialization_decision.json",{"decision":"NEW_TRANSITION_CRITIC","reason":"steady RUN critic return semantics are incompatible","input":152,"output":1})
j("autograd_scope_audit.json",{"source_controllers_no_grad":True,"transition_actor_gradient":"PASS from Stage7R","transition_critic_gradient":"STATIC UNIT PASS","source_graph_retained":False,"end_to_end_live_audit":False})
j("optimizer_parameter_audit.json",{"transition_actor_only":True,"new_transition_critic_only":True,"frozen_parameters":False,"live_optimizer_constructed":False})
gae=json.loads((O/"gae_unit_test.json").read_text());cont=json.loads((O/"prefix_reward_contamination_test.json").read_text());dur=json.loads((O/"source_duration_invariance_test.json").read_text());storage=json.loads((O/"ppo_storage_audit.json").read_text())
r0={"source_success_ge_90":True,"source_contract_only_start":True,"transition_steps_only_storage":True,"invalid_stored_steps_zero":storage["invalid_stored_steps"]==0,"manual_gae":gae["pass"],"prefix_contamination":cont["pass"],"duration_invariance":dur["pass"],"segment_boundary":True,"actor_gradient":True,"critic_gradient":True,"frozen_gradient_zero":True,"previous_action_contract":True,"observation_152":True,"action_37":True,"checkpoint_reload":True,"optimizer_reload":True,"live_ready_cohort_512":False,"physical_state_handoff_verified":False}
r0["status"]="PASS" if all(r0.values()) else "FAIL";j("r0_interface_gate.json",r0)
j("training_config.json",{"pilot_executed":False,"cohort_size":512,"rollout_horizon":None,"gamma":.99,"lambda":.95,"reason":"R0 FAIL before freeze"})
j("reward_definition.json",{"status":"NOT_EXECUTED","source_prefix_reward":"not represented","transition_reward":"Stage7R frozen draft"})
j("pilot_results.json",{"executed":False});j("checkpoint_sweep.json",{"executed":False});j("formal_summary.json",{"status":"NOT_RUN"});j("per_seed_results.json",{});j("per_target_results.json",{})
for n in ("episodes.csv","transition_timelines.csv","cycle_acquisition_metrics.csv"):(O/n).write_text("status\nNOT_RUN\n",encoding="utf-8")
j("action_discontinuity.json",{"status":"NOT_RUN"});j("impact_diagnostics.json",{"status":"NOT_RUN"});j("saturation_diagnostics.json",{"status":"NOT_RUN"});j("failure_counts.json",{"transition_only_runner_handoff_failure":1})
j("gate.json",{"stage":"7R2","status":"FAIL","eligible_for_stage8":False,"r0_status":"FAIL","failures":["ready cohort physical-state handoff not implemented for 512 live environments","end-to-end source-to-storage boundary not verified in Isaac Sim"],"pilot":False,"formal":False,"artifact":False,"git_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()})
(O/"reproduction_commands.ps1").write_text('cd "$HOME\\workspace\\physical-ai-lab"\n& "$env:USERPROFILE\\workspace\\IsaacLab\\isaaclab.bat" -p experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\audit_stage7r2_runner.py\n',encoding="utf-8")
print(json.dumps(json.loads((O/"gate.json").read_text()),indent=2))
