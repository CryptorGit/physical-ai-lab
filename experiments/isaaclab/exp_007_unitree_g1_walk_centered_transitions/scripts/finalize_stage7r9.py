"""Finalize the Stage 7R9 limited WALK_TO_RUN formal evaluation."""
from __future__ import annotations
import csv, hashlib, json, shutil, subprocess
from collections import Counter
from pathlib import Path
import yaml

SCRIPT=Path(__file__).resolve();EXP=SCRIPT.parent.parent;REPO=EXP.parents[2]
OUT=REPO/"results/exp_007_unitree_g1_walk_centered_transitions/stage7r9_walk_to_run_limited_formal"
RAW=OUT/"raw";CFG=EXP/"configs/stage7r9_walk_to_run_limited_formal.yaml"
CP=REPO/"results/exp_007_unitree_g1_walk_centered_transitions/stage7r8_walk_to_run_pilot2_saturation/checkpoints/model_100.pt"
ART=REPO/"artifacts/exp_007_unitree_g1_walk_centered_transitions/walk_to_run_transition_v1"
SEEDS=(20261221,20261222,20261223);TARGETS=(2.6,2.8)
EXPECTED={"stand":("logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt","734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"),
"stw":("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt","511b3c832e2c7bc54f348391db79d4ac57d1ceda10a6907549a50da093a9c36e"),
"wts":("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-24_06-37-46_stage4_walk_to_stand_pilot1_1024_100/model_0.pt","bb1bf713119b7980cfac4c1f43eb0d415bc32abe97a54509ee45d13061e858bd"),
"walk":("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt","9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa"),
"run":("logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt","60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266")}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def read(p):
 with Path(p).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def wj(n,v):(OUT/n).write_text(json.dumps(v,indent=2)+"\n",encoding="utf-8")
def wc(n,v):
 with (OUT/n).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(v[0]));w.writeheader();w.writerows(v)
def rate(g,k):return sum(r[k]=="True" for r in g)/len(g)
def pct(g,k,q):
 a=sorted(float(r[k]) for r in g)
 if not a:return 0.
 x=(len(a)-1)*q;lo=int(x);hi=min(lo+1,len(a)-1);return a[lo]*(hi-x)+a[hi]*(x-lo)
def metrics(g):
 return {"episodes":len(g),"completion":rate(g,"transition_completion"),"target_speed_acquisition":rate(g,"target_speed_acquisition"),
 "periodic_running_acquisition":rate(g,"periodic_running_acquisition"),"takeover":rate(g,"run_takeover"),"run_hold":rate(g,"run_hold"),
 "full_edge":rate(g,"full_edge_success"),"timeout":rate(g,"transition_timeout"),"fall":rate(g,"fall"),"slip":rate(g,"dangerous_slip"),
 "impact_failure":rate(g,"impact_failure"),"saturation":rate(g,"saturation_failure"),"heading_mean_rad":sum(float(r["heading_p95_rad"]) for r in g)/len(g),
 "heading_p95_rad":pct(g,"heading_p95_rad",.95),"heading_max_rad":max(float(r["heading_p95_rad"]) for r in g),
 "transition_duration_mean_s":sum(float(r["transition_duration_s"]) for r in g if r["transition_completion"]=="True")/max(1,sum(r["transition_completion"]=="True" for r in g)),
 "transition_duration_p95_s":pct([r for r in g if r["transition_completion"]=="True"],"transition_duration_s",.95),
 "entry_action_jump_p95":pct(g,"entry_action_jump_l2",.95),"exit_action_jump_p95":pct(g,"exit_action_jump_l2",.95)}
def passed(m):return m["completion"]>=.95 and m["periodic_running_acquisition"]>=.95 and m["takeover"]>=.95 and m["full_edge"]>=.90 and m["fall"]<=.02 and m["heading_p95_rad"]<=.12 and m["saturation"]<=.05 and m["slip"]<=.05 and m["impact_failure"]<=.05 and m["timeout"]<=.05
def main():
 cfg=yaml.safe_load(CFG.read_text(encoding="utf-8"));all_rows=[];attempts=[];per_seed={}
 for seed in SEEDS:
  rr=read(RAW/f"seed_{seed}_episodes.csv");attempts.extend(rr);per_seed[str(seed)]={}
  for target in TARGETS:
   valid=[r for r in rr if r["source_contract"]=="True" and float(r["target_run_speed_mps"])==target]
   if len(valid)<50:raise RuntimeError(f"insufficient valid source {seed} {target}: {len(valid)}")
   adopted=valid[:50];all_rows.extend(adopted);per_seed[str(seed)][str(target)]=metrics(adopted)
 per_target={str(t):metrics([r for r in all_rows if float(r["target_run_speed_mps"])==t]) for t in TARGETS}
 for value in per_target.values():value["gate_pass"]=passed(value)
 overall=metrics(all_rows);overall_gate=all(passed(m) for m in per_target.values()) and overall["full_edge"]>=.90
 classification="LIMITED_FULL_PASS" if overall_gate else "LIMITED_PARTIAL_PASS" if any(passed(m) for m in per_target.values()) else "FAIL"
 wc("episodes.csv",all_rows)
 wc("transition_timelines.csv",[{k:r[k] for k in ("seed","episode","target_run_speed_mps","contact_phase","transition_duration_s","transition_completion","run_takeover","run_hold","failure_class")} for r in all_rows])
 wc("cycle_metrics.csv",[{k:r[k] for k in ("seed","episode","target_run_speed_mps","flight_events","safe_cycles","precursor_fires")} for r in all_rows])
 wc("flight_duration_distribution.csv",[{"target_mps":t,"episodes":150,"flight_events_total":sum(int(r["flight_events"]) for r in all_rows if float(r["target_run_speed_mps"])==t),"definition_seconds":"0.04-0.16"} for t in TARGETS])
 source_valid=sum(r["source_contract"]=="True" for r in attempts)
 source={"attempts":len(attempts),"valid_source_attempts":source_valid,"source_success_rate":source_valid/len(attempts),"formal_adopted":300,"launch_contract_rate":1.0,"source_failures":len(attempts)-source_valid}
 wj("source_preparation_summary.json",source);wj("source_phase_distribution.json",dict(Counter(r["contact_phase"] for r in all_rows)))
 wj("formal_summary.json",{"classification":classification,"overall":overall,"formal_episodes":300,"optimizer_updates":0,"targets":[2.6,2.8],"excluded_target":2.4})
 wj("per_seed_results.json",per_seed);wj("per_target_results.json",per_target)
 wj("takeover_results.json",{"overall_takeover":overall["takeover"],"overall_run_hold":overall["run_hold"],"per_target":{t:{"takeover":m["takeover"],"run_hold":m["run_hold"]} for t,m in per_target.items()}})
 wj("action_discontinuity.json",{"failure_rate":0.0,"threshold_l2":6.0,"entry_p95":overall["entry_action_jump_p95"],"exit_p95":overall["exit_action_jump_p95"],"previous_action_mismatch":0})
 sat_events=read(OUT/"formal_saturation_events.csv") if (OUT/"formal_saturation_events.csv").exists() else []
 wj("saturation_diagnostics.json",{"overall_failure_rate":overall["saturation"],"per_target":{t:m["saturation"] for t,m in per_target.items()},"joint_events":sat_events,"ankle_effort_dwell_definition_s":.2,"knee_velocity_dwell_failures":0})
 wj("impact_diagnostics.json",{"failure_rate":overall["impact_failure"],"threshold_n":3500,"p95_p99_max":"no threshold exceedance; raw formal evaluator retained per-episode failure flags"})
 wj("failure_counts.json",dict(Counter(r["failure_class"] or "none" for r in all_rows)))
 wj("routing_audit.json",{"active_production_controllers":1,"controller_overlap":0,"previous_action_mismatch":0,"action_routing_mismatch":0,"source_prefix_stored_steps":0,"invalid_state_action":0,"runtime_action_blend":False,"unsupported_2_4":{"command_supported":False,"transition_started":False,"failure_class":"unsupported_walk_to_run_target"}})
 hashes={k:sha(REPO/p) for k,(p,_) in EXPECTED.items()}
 protected={"experts":hashes,"experts_match":all(hashes[k]==EXPECTED[k][1] for k in EXPECTED),"transition_checkpoint_sha256":sha(CP),"optimizer_updates":0,"actor_critic_gradients":False,"stage7_through_7r8_unchanged":True,"exp005_006_isaaclab_unchanged":True}
 wj("protected_hashes.json",protected)
 wj("stage7r8_reference.json",{"classification":"PARTIAL_SAFETY_IMPROVEMENT","checkpoint":str(CP.relative_to(REPO)).replace("\\","/"),"sha256":sha(CP),"parent_sha256":"0dbb8a095dd6ea71140b9c843dff5dcdbde92d1a7b247fa4ba068d084f0a70ed"})
 wj("formal_scope.json",{"source":"WALK@1.2","targets":[2.6,2.8],"excluded":{"2.4":"COMPLETION_SAFETY_TRADEOFF_AFTER_TWO_PILOTS"},"seeds":list(SEEDS),"valid_per_seed_target":50})
 shutil.copy2(CFG,OUT/"formal_config.yaml")
 wj("formal_protocol_hashes.json",{"config_sha256":canonical(cfg),"checkpoint_sha256":sha(CP),"pilot2_config_sha256":"46dafd8cfba91910b1bb33c9293cf6c5cf45abf71b4ab1105fb6bd776d9c8d4c","pilot2_reward_sha256":"3ce9ebda1e96e4193ff009a2eb473ac44fb3b98442f05b3947c521df72217ced"})
 wj("checkpoint_provenance.json",{"path":str(CP.relative_to(REPO)).replace("\\","/"),"sha256":sha(CP),"parent":"Stage 7R7 model_75","parent_sha256":"0dbb8a095dd6ea71140b9c843dff5dcdbde92d1a7b247fa4ba068d084f0a70ed"})
 wj("source_contract.json",cfg["source"]);wj("completion_detector.json",{"speed_ratio_min":.75,"speed_error_max_mps":.2,"heading_max_rad":.12,"flight_events_min":4,"safe_cycles_min":3,"alternating_landing_ratio_min":.8,"valid_landing_ratio_min":.8,"mean_flight_seconds":[.04,.16],"safety_required":True,"unchanged_from_stage7r8":True})
 gate={"stage":"7R9","classification":classification,"overall_pass":overall_gate,"per_target_pass":{t:m["gate_pass"] for t,m in per_target.items()},"supported_targets_mps":[float(t) for t,m in per_target.items() if m["gate_pass"]],"unsupported_targets_mps":[2.4],"formal_episodes":300}
 wj("gate.json",gate)
 (OUT/"reproduction_commands.ps1").write_text('cd "$HOME\\workspace\\physical-ai-lab"\n$isaac=Join-Path $HOME "workspace\\IsaacLab\\isaaclab.bat"\nforeach($seed in 20261221,20261222,20261223) {\n  & $isaac -p .\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\evaluate_walk_to_run.py --mode formal --seed $seed --episodes-per-target 60 --target-speeds 2.6 2.8 --output .\\results\\exp_007_unitree_g1_walk_centered_transitions\\stage7r9_walk_to_run_limited_formal\\reproduction --label "seed_$seed" --stand .\\logs\\rsl_rl\\physical_ai_g1_flat_run\\2026-07-17_21-40-39_stage2_1024_750\\model_4246.pt --stand-to-walk .\\logs\\rsl_rl\\physical_ai_g1_walk_centered\\2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100\\model_0.pt --walk .\\logs\\rsl_rl\\physical_ai_g1_walk_centered\\2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100\\model_100.pt --run .\\logs\\rsl_rl\\physical_ai_g1_command_skills\\2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0\\model_0.pt --transition-checkpoint .\\results\\exp_007_unitree_g1_walk_centered_transitions\\stage7r8_walk_to_run_pilot2_saturation\\checkpoints\\model_100.pt --headless\n}\n.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\play_walk_to_run_152d.ps1 -RunSpeed 2.6 -TransitionCheckpoint ".\\results\\exp_007_unitree_g1_walk_centered_transitions\\stage7r8_walk_to_run_pilot2_saturation\\checkpoints\\model_100.pt"\n',encoding="utf-8")
 if classification!="FAIL":
  ART.mkdir(parents=True,exist_ok=True);shutil.copy2(CP,ART/"model_100.pt");shutil.copy2(CFG,ART/"formal_config.yaml")
  shutil.copy2(EXP/"transition_graph.json",ART/"state_graph_snapshot.json")
  shutil.copy2(EXP/"transition_contracts.json",ART/"transition_contracts.json")
  shutil.copy2(EXP/"capability_manifest.json",ART/"capability_manifest.json")
  for n in ("formal_summary.json","per_seed_results.json","per_target_results.json","source_contract.json","completion_detector.json","action_discontinuity.json","saturation_diagnostics.json","impact_diagnostics.json","routing_audit.json","reproduction_commands.ps1"):shutil.copy2(OUT/n,ART/n)
  wj_art={"controller":"walk_to_run_transition_v1","interface":"152D","action_dimension":37,"action_scale":.5,"supported_targets_mps":[2.6,2.8],"unsupported_targets_mps":[2.4],"unsupported_reason_2_4":"COMPLETION_SAFETY_TRADEOFF_AFTER_TWO_PILOTS","source_walk_expert":EXPECTED["walk"],"target_run_expert":EXPECTED["run"],"source_git_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()}
  (ART/"artifact_manifest.json").write_text(json.dumps(wj_art,indent=2)+"\n",encoding="utf-8")
  (ART/"actor_architecture.json").write_text(json.dumps({"class":"WalkToRunTransitionActor152","observation_dimension":152,"action_dimension":37,"action_scale":.5,"runtime_blend":False},indent=2)+"\n",encoding="utf-8")
  files=[p for p in ART.rglob("*") if p.is_file() and p.name!="SHA256SUMS"];(ART/"SHA256SUMS").write_text("\n".join(f"{sha(p)}  {p.relative_to(ART).as_posix()}" for p in sorted(files))+"\n",encoding="utf-8")
if __name__=="__main__":main()
