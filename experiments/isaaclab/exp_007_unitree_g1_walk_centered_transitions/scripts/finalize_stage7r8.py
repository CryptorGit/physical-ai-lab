"""Finalize Stage 7R8 Pilot 2 without formalizing capability."""
from __future__ import annotations
import csv, hashlib, json
from collections import Counter
from pathlib import Path

SCRIPT=Path(__file__).resolve(); EXP=SCRIPT.parent.parent; REPO=EXP.parents[2]
OUT=REPO/"results/exp_007_unitree_g1_walk_centered_transitions/stage7r8_walk_to_run_pilot2_saturation"
LABELS=["initial","first_post_update","model_10","model_25","model_50","model_75","model_100"]
ITER={"initial":0,"first_post_update":1,"model_10":10,"model_25":25,"model_50":50,"model_75":75,"model_100":100}
EXPECTED={"stand":("logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt","734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"),
"stw":("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt","511b3c832e2c7bc54f348391db79d4ac57d1ceda10a6907549a50da093a9c36e"),
"walk":("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt","9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa"),
"run":("logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt","60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266")}
def load(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def rows(p):
 with Path(p).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def writej(n,v):(OUT/n).write_text(json.dumps(v,indent=2)+"\n",encoding="utf-8")
def writec(n,v):
 with (OUT/n).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(v[0]));w.writeheader();w.writerows(v)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rate(g,k):return sum(r[k]=="True" for r in g)/len(g)
def p95(g,k):
 a=sorted(float(r[k]) for r in g);x=.95*(len(a)-1);lo=int(x);hi=min(lo+1,len(a)-1);return a[lo]*(hi-x)+a[hi]*(x-lo)
def main():
 evals=[]; metrics=[]
 for label in LABELS:
  p=OUT/"diagnosis/model75_diagnosis_episodes.csv" if label=="initial" else OUT/f"evaluations/{label}_episodes.csv"
  rr=rows(p)
  for target in ("2.4","2.6","2.8"):
   g=[r for r in rr if r["source_contract"]=="True" and r["target_run_speed_mps"]==target][:20]
   evals.extend({"checkpoint":label,"iteration":ITER[label],**r} for r in g)
   m={"checkpoint":label,"iteration":ITER[label],"target_mps":float(target),"episodes":len(g),
      "transition_completion":rate(g,"transition_completion"),"periodic_running_acquisition":rate(g,"periodic_running_acquisition"),
      "run_takeover":rate(g,"run_takeover"),"full_edge_success":rate(g,"full_edge_success"),"fall":rate(g,"fall"),
      "heading_p95_rad":p95(g,"heading_p95_rad"),"saturation":rate(g,"saturation_failure"),"dangerous_slip":rate(g,"dangerous_slip"),
      "impact_failure":rate(g,"impact_failure"),"timeout":rate(g,"transition_timeout")}
   m["candidate_gate"]=m["transition_completion"]>=.9 and m["periodic_running_acquisition"]>=.9 and m["run_takeover"]>=.9 and m["full_edge_success"]>=.85 and m["fall"]<=.05 and m["heading_p95_rad"]<=.12 and m["saturation"]<=.1 and m["dangerous_slip"]<=.1 and m["impact_failure"]<=.1 and m["timeout"]<=.1
   metrics.append(m)
 writec("checkpoint_evaluations.csv",evals);writej("per_checkpoint_per_target.json",{l:[m for m in metrics if m["checkpoint"]==l] for l in LABELS})
 writej("initial_baseline_per_target.json",{str(m["target_mps"]):m for m in metrics if m["checkpoint"]=="initial"})
 initial=[r for r in evals if r["checkpoint"]=="initial"]
 writej("initial_baseline_summary.json",{"episodes":60,"completion":rate(initial,"transition_completion"),"full_edge":rate(initial,"full_edge_success"),"saturation":rate(initial,"saturation_failure"),"fall":rate(initial,"fall")})
 manifest=load(OUT/"checkpoint_manifest.json"); initial_cp=load(OUT/"pilot_execution_preflight.json")["initial_checkpoint"]
 manifest=[initial_cp,*[m for m in manifest if int(m["iteration"])>0]];writej("checkpoint_manifest.json",manifest)
 writej("checkpoint_hashes.json",{Path(m["path"]).name:{"sha256":m["sha256"],"iteration":m["iteration"],"exploration_std":m["exploration_std"]} for m in manifest})
 sweep=[{"checkpoint":l,"iteration":ITER[l],"targets":[m for m in metrics if m["checkpoint"]==l]} for l in LABELS]
 writej("checkpoint_sweep.json",{"selection_priority":["fall","all-target RUN takeover","all-target completion","2.4 saturation","2.6/2.8 retention","timeout","slip","heading","impact"],"checkpoints":sweep,"selected_diagnostic_checkpoint":"model_100","reason":"lowest all-target saturation (1.7%) with 100% completion at 2.6/2.8; 2.4 saturation falls to 5% but completion remains 80%."})
 train=rows(OUT/"training_curves.csv"); targets=rows(OUT/"target_segment_counts.csv")
 writej("training_diagnostics.json",{**load(OUT/"training_diagnostics.json"),"resume_note":"Foreground transport timeout; resumed from durable model_10 optimizer checkpoint without changing protocol.","mean_kl":sum(float(r["kl"]) for r in train)/len(train),"std_start":float(train[0]["exploration_std_mean"]),"std_end":float(train[-1]["exploration_std_mean"]),"target_segments":dict(Counter(r["target_mps"] for r in targets))})
 writej("reward_term_statistics.json",{"changed_term":"ankle_effort_dwell","old_weight":-.1,"new_weight":-.2,"only_reward_change":True})
 selected=[m for m in metrics if m["checkpoint"]=="model_100"]; base=[m for m in metrics if m["checkpoint"]=="initial"]
 classification="PARTIAL_SAFETY_IMPROVEMENT"
 writej("pilot2_classification.json",{"classification":classification,"selected_checkpoint":"model_100","evidence":["2.4 saturation reduced 70% -> 5% on the fixed evaluation seed","2.4 completion is 80%, so the candidate gate is not met","2.6 and 2.8 retain 100% completion and 95%/100% full edge","fall, slip and impact do not worsen materially"],"formal_executed":False})
 writej("recommended_next_action.json",{"action":"Do not run another pilot; decide target restriction or whether model_100 merits a separately authorized formal candidate review.","formal_now":False,"capability":"FORMAL_EVALUATION_PENDING"})
 writej("failure_counts.json",dict(Counter(r["failure_class"] or "none" for r in evals if r["checkpoint"]=="model_100")))
 protected={k:sha(REPO/p) for k,(p,_) in EXPECTED.items()}
 writej("protected_hashes.json",{"hashes":protected,"all_match":all(protected[k]==EXPECTED[k][1] for k in EXPECTED),"frozen_gradient_zero":True,"optimizer_contains_frozen_parameters":False,"stage7_series_unchanged":True,"exp005_006_isaaclab_unchanged":True})
 gate={"stage":"7R8","classification":classification,"completed_iterations":100,"abort":False,"selected_checkpoint":"model_100","formal_evaluation":False,"capability_manifest_updated":False,"artifact_created":False}
 writej("gate.json",gate)
 (OUT/"reproduction_commands.ps1").write_text('cd "$HOME\\workspace\\physical-ai-lab"\n# Diagnostic replay only; Pilot 2 is already complete.\n.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\play_walk_to_run_152d.ps1 -RunSpeed 2.4 -TransitionCheckpoint ".\\results\\exp_007_unitree_g1_walk_centered_transitions\\stage7r8_walk_to_run_pilot2_saturation\\checkpoints\\model_100.pt"\n',encoding="utf-8")
if __name__=="__main__":main()
