"""Finalize the bounded Stage 7 failure without creating production capability."""
import csv, hashlib, json, subprocess
from collections import Counter
from pathlib import Path
H=Path(__file__).resolve();EXP=H.parent.parent;REPO=EXP.parents[2];O=REPO/"results/exp_007_unitree_g1_walk_centered_transitions/stage7_walk_to_run";O.mkdir(parents=True,exist_ok=True)
def j(name,x):(O/name).write_text(json.dumps(x,indent=2)+"\n",encoding="utf-8")
def rows(name):
 with (O/name).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
base=json.loads((O/"direct_switch_baseline_summary.json").read_text());ramp=json.loads((O/"direct_switch_ramp_1p4_summary.json").read_text())
br=rows("direct_switch_baseline_episodes.csv"); ov=rows("direct_switch_baseline_overlap.csv")
protected={"WALK":"9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa","RUN_LOW":"60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266","STAND":"734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621","STAND_TO_WALK":"511b3c832e2c7bc54f348391db79d4ac57d1ceda10a6907549a50da093a9c36e"}
j("source_walk_contract.json",{"state":"WALK","expert":"walk_steady_state_expert_v1","command_mps":1.2,"speed_error_max":.2,"heading_error_max":.12,"hold_s":1.0,"safety_required":True})
j("target_run_contract.json",{"state":"RUN_LOW","commands_mps":[2.4,2.6,2.8],"periodic_definition":"Stage 6","hold_s":5})
j("walk_prepare_run_diagnostic.json",{"status":"NOT_FORMALIZED","planner_phase":"WALK_PREPARE_RUN","required_before_future_retry":True,"source_commands_mps":[.6,.8,1.]})
j("overlap_audit.json",{"episodes":len(ov),"action_l2_mean":sum(float(r["walk_run_action_l2"]) for r in ov)/len(ov),"rows":"direct_switch_baseline_overlap.csv"})
j("contact_phase_audit.json",dict(Counter(r["contact_phase"] or "unknown" for r in br if r["source_contract"]=="True")))
j("direct_switch_baseline.json",base)
j("controller_classification.json",{"classification":"DIRECT_SWITCH_FAIL","parameter_free_ramp_1p4":ramp,"learned_controller":"R0_WIRING_BLOCKED","reason":"No existing trainable 152-dimensional transition action term can start from frozen WALK occupancy without violating strict RUN actor/interface provenance."})
j("completion_detector.json",{"frozen_before_audit":True,"speed_ratio_min":.75,"speed_error_max":.2,"heading_max":.12,"flight_events_min":4,"safe_cycles_min":3,"alternation_min":.8,"valid_landing_min":.8,"mean_flight_s":[.04,.16],"timeout_s":5})
j("training_config.json",{"parent":"RUN_LOW model_0","optimizer":"RESET_REQUIRED","R0":{"status":"FAIL","updates":0,"reason":"152D G1CommandResidualActor transition task from WALK occupancy not available"},"pilot_1":{"executed":False},"pilot_2":{"executed":False}})
j("reward_definition.json",{"status":"SPECIFIED_NOT_EXECUTED","reuse":["safe-flight precursor","periodic completion","speed/heading/upright","impact/slip/saturation/action-rate"]})
j("endpoint_alignment.json",{"runtime_blend":False,"source":"WALK action regularization planned","target":"RUN action regularization planned","executed":False})
j("pilot_results.json",{"direct_immediate":base,"parameter_free_ramp_1p4":ramp,"learned_pilots":[]})
j("checkpoint_sweep.json",{"status":"NOT_AVAILABLE","selected_checkpoint":None})
j("formal_summary.json",{"status":"NOT_RUN_NO_ELIGIBLE_CANDIDATE","performance_claim":False})
j("per_seed_results.json",{});j("per_target_results.json",base["per_target"])
(O/"episodes.csv").write_text((O/"direct_switch_baseline_episodes.csv").read_text(),encoding="utf-8")
for name in ("transition_timelines.csv","cycle_acquisition_metrics.csv"):
 (O/name).write_text("status\nNOT_AVAILABLE\n",encoding="utf-8")
j("action_discontinuity.json",{"failure_rate":base["action_discontinuity_failure_rate"],"entry_action_l2_mean":sum(float(r["entry_action_jump_l2"]) for r in br if r["source_contract"]=="True")/base["valid_sources"],"exit":"not reached reliably"})
j("impact_diagnostics.json",{"failure_rate":base["impact_failure_rate"]})
j("saturation_diagnostics.json",{"failure_rate":base["saturation_rate"]})
j("takeover_results.json",{"rate":base["run_takeover_rate"]})
j("failure_counts.json",dict(Counter(r["failure_class"] or "none" for r in br)))
pre=json.loads((O/"routing_preflight.json").read_text());pre["protected_hashes"]=protected;j("routing_preflight.json",pre)
j("gate.json",{"stage":7,"status":"FAIL","eligible_for_stage8":False,"failures":["direct switch pilot gate failed","no valid learned transition R0 task/checkpoint"],"direct_switch":base,"formal":"NOT_RUN","supported_targets":[],"protected_hashes":protected,"git_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()})
(O/"reproduction_commands.ps1").write_text('cd "$HOME\\workspace\\physical-ai-lab"\n.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\evaluate_walk_to_run.ps1 -Mode baseline -Seed 20261102 -Label direct_switch_baseline -EpisodesPerTarget 10\n.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\evaluate_walk_to_run.ps1 -Mode baseline -Seed 20261103 -Label direct_switch_ramp_1p4 -EpisodesPerTarget 10\n',encoding="utf-8")
print(json.dumps(json.loads((O/"gate.json").read_text()),indent=2))
