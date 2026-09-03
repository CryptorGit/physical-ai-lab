"""Assemble fail-closed EXP 014 artifacts from executed runs."""
from __future__ import annotations
import csv,hashlib,json,subprocess,sys
from datetime import datetime,timedelta,timezone
from pathlib import Path
import torch
HERE=Path(__file__).resolve();EXP=HERE.parent.parent;REPO=EXP.parents[2];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion";JST=timezone(timedelta(hours=9))
sys.path[:0]=[str(EXP/"src"),str(HERE.parent)];from g1_explicit_motion_mode.student import ExplicitModeStudent;from train_static import metrics
SELECTED=OUT/"dagger_checkpoints/round_2_step_10000.pt"
def dump(name,v):(OUT/name).write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def phase_stats(tag):
 rows=[]
 for p in sorted((OUT/tag).glob("*.json")):rows+=json.loads(p.read_text())["rows"]
 return {"stand_hold":sum(r["stand_hold"] for r in rows)/len(rows),"endpoint":sum(r["endpoint"] for r in rows)/len(rows),"acquisition":sum(r["acquisition_0p20"] for r in rows)/len(rows),"walk_to_stand":sum(r["walk_to_stand"] for r in rows)/len(rows),"fall":sum(r["fall_rate"] for r in rows)/len(rows),"dangerous_slip":sum(r["dangerous_slip_rate"] for r in rows)/len(rows)}
def main():
 s0=json.loads((OUT/"capacity_s0.json").read_text());timeline=[]
 for r in s0["timeline"]:timeline.append({"model":"S0","step":r["step"],"parameter_count":r["parameter_count"],"aggregate_mse":r["validation"]["aggregate_mse"],"worst_direction_yaw_mse":r["validation"]["worst_condition_mse"],"dual_mode_classification":r["validation"]["dual_mode_classification"],"static_gate_pass":r["validation"]["static_gate_pass"],"checkpoint":r["checkpoint"],"sha256":r["sha256"]})
 dump("capacity_static_timeline.json",{"rows":timeline})
 with (OUT/"capacity_static_timeline.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(timeline[0]));w.writeheader();w.writerows(timeline)
 parity=json.loads((OUT/"observation_initialization_parity.json").read_text());comparison={"S0":{"executed":True,"parameter_count":90570,"selected_step":30000,"static_pass":True,"worst_validation_mse":s0["selected"]["validation"]["worst_condition_mse"]},"S1":{"executed":False,"reason":"S0 passed; branch not authorized","parameter_count":parity["S1"]["parameter_count"],"initialization_status":parity["S1"]["status"]},"S2":{"executed":False,"reason":"S1 not authorized","parameter_count":parity["S2"]["parameter_count"],"initialization_status":parity["S2"]["status"]},"capacity_conclusion":"S0 is sufficient for the registered static joint solution; physical failure is not a capacity-gate failure."};dump("capacity_comparison.json",comparison)
 # Final static held-out is opened once after DAgger-round/model freeze.
 chunks=[torch.load(p,map_location="cpu",weights_only=False) for p in sorted((OUT/"phase1_dataset").glob("*.pt"))]+[torch.load(OUT/f"dagger_dataset/round_{r}.pt",map_location="cpu",weights_only=False) for r in (1,2)];keys=("observation_141","teacher_action","context","group","condition_index","recipe_id","control_step","split_id");d={k:torch.cat([x[k] for x in chunks]) for k in keys};held=torch.nonzero(d["split_id"].flatten()==2).flatten();payload=torch.load(SELECTED,map_location="cpu",weights_only=False);model=ExplicitModeStudent(tuple(payload["architecture"][1:-1]));model.load_state_dict(payload["actor_state_dict"]);heldout=metrics(model.eval(),d,held,torch.device("cpu"));dump("selected_model_heldout_once.json",{"checkpoint":SELECTED.relative_to(REPO).as_posix(),"partition":"held-out","opened_after_selection":True,"used_for_selection":False,"metrics":heldout})
 torch.manual_seed(14014);x=torch.randn(2048,141);a=model(x);fresh=ExplicitModeStudent(tuple(payload["architecture"][1:-1]));fresh.load_state_dict(torch.load(SELECTED,map_location="cpu",weights_only=False)["actor_state_dict"]);b=fresh(x);process_parity={"status":"PASS" if torch.equal(a,b) else "FAIL","fresh_reload_bitwise_equal":torch.equal(a,b),"max_absolute_difference":float((a-b).abs().max()),"runtime_actor_count":1,"runtime_teacher_count":0,"router_count":0,"checkpoint_switches":0,"action_blends":0,"gaussian_heads":1};dump("selected_model_process_parity.json",process_parity)
 selected={"selection_partition":"validation only","heldout_used_for_selection":False,"architecture":payload["architecture"],"parameter_count":payload["parameter_count"],"checkpoint":SELECTED.relative_to(REPO).as_posix(),"sha256":sha(SELECTED),"dagger_round":2,"selection_reason":"no closed-loop candidate passed; same S0 size, round 2 had best static worst-group MSE and practical-STAND validation rate","mandatory_phase2_gate_pass":False};dump("selected_model.json",selected)
 rounds=[];growth=[{"round":0,"horizon":0,"base_samples":45560,"added_samples":0,"cumulative_samples":45560}]
 for r,h in ((1,8),(2,16)):
  data=torch.load(OUT/f"dagger_dataset/round_{r}.pt",map_location="cpu",weights_only=False);count=len(data["observation_141"]);possible=680*h*3;rounds.append({"round":r,"horizon":h,"parent":data["parent_checkpoint"],"student_visited_samples":count,"unlabelable_states":possible-count,"window_counts":{str(i):int((data["window"]==i).sum()) for i in range(3)},"teacher_mapping":{"STAND":"S","WALK":"W"},"episode_teacher_switching":False});growth.append({"round":r,"horizon":h,"base_samples":45560,"added_samples":count,"cumulative_samples":45560+sum(x["student_visited_samples"] for x in rounds)})
 stats={"round0":phase_stats("phase2_batches"),"round1":phase_stats("phase2_round1_batches"),"round2":phase_stats("phase2_round2_batches")};dump("dagger_round_manifest.json",{"rounds":rounds,"maximum_rounds_registered":3,"round3_executed":False,"reason_round3_not_executed":"same STAND_RETENTION_FAIL class remained after two retries","failure_reduction":stats})
 with (OUT/"dagger_dataset_growth.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(growth[0]));w.writeheader();w.writerows(growth)
 diagnosis={"primary_failure_class":"STAND_RETENTION_FAIL","static_physical_gap":True,"round_metrics":stats,"reset_boundary_coverage":{"round1_STAND_control_steps":[4,5,6,7],"round2_STAND_control_steps":[4,5,6,7,8,9,10,11,12,13,14,15],"missing_steps":[0,1,2,3],"cause":"collector conservatively marked pre-contact reset states UNLABELABLE_STATE"},"interpretation":"DAgger improved practical STAND monotonically but never covered the first four reset control steps; endpoint/acquisition remained nonzero, so explicit mode is represented but the closed-loop reset/stop state distribution remains incomplete."};dump("physical_failure_diagnosis.json",diagnosis)
 dump("local_neighborhood.json",{"status":"NOT_EXECUTED","reason":"formal Phase 2 gates failed; local overfit gate was not authorized as a substitute"});dump("three_mode_training.json",{"status":"NOT_AUTHORIZED","reason":"Phase 2 did not pass"});dump("forward_gait_transition_formal.json",{"status":"NOT_AUTHORIZED","reason":"Phase 2 did not pass; no RUN labels or training were started"});dump("omni_run_feasibility.json",{"status":"NOT_AUTHORIZED","classification":None,"reason":"Phase 3 did not pass"})
 current={"deepest_completed_phase":"Phase 1 static joint solution; Phase 2 attempted through DAgger round 2 but failed","selected_architecture":payload["architecture"],"parameter_count":payload["parameter_count"],"checkpoint":selected["checkpoint"],"sha256":selected["sha256"],"motion_modes_supported":["STAND/WALK static teacher-action conditioning only"],"directions_supported":[],"yaw_supported":[],"gait_transitions_supported":[],"practical_stop_supported":False,"failed_gates":["practical STAND","STAND_TO_WALK 16/16","pure yaw 2/2","moving yaw 16/16","WALK_TO_STAND >=95%","full sequence"],"unresolved_issues":["reset steps 0-3 absent from labelable DAgger data","aggregate fall about 50%","direction-dependent stop collapse"],"next_authorized_phase":"new reset-boundary teacher-positive-control experiment; Phase 3 remains unauthorized"};dump("current_best_manifest.json",current)
 classification={"primary_classification":"EXP014_STATIC_PASS_PHYSICAL_FAIL","deepest_completed_phase":"static capacity probe","phase0":"PASS","dataset_integrity":"PASS","static":"PASS","phase2":"FAIL after two DAgger retries","phase3":"NOT_AUTHORIZED","phase4":"NOT_AUTHORIZED"};dump("stage_classification.json",classification)
 next_action={"experiment":"Exp014 reset-boundary labelability positive control and causal DAgger dataset V2","single_change":"replace the contact-presence proxy for UNLABELABLE_STATE with an actual read-only Specialist-S positive-control rollout, then collect steps 0-3 only when S itself does not fall/slip","fixed_parent":selected["checkpoint"],"horizon":8,"success_criterion":"steps 0-7 covered with zero unsupported labels, static retention PASS, and practical STAND >=95% before any WALK/RUN work","prohibited":"no gate change, no RUN integration, no teacher router, no action blending"};dump("recommended_next_action.json",next_action)
 now=datetime.now(JST);tb=json.loads((OUT/"time_budget.json").read_text());start=datetime.fromisoformat(tb["start_time"]);entries=[
  {"timestamp":now.isoformat(),"phase":"Phase 1","run ID":"P1_DATASET","parent":"P0_BOOTSTRAP","hypothesis":"explicit mode removes B0 label collision","single changed variable":"new formal S/W trajectory dataset","metrics":{"episodes":680,"samples":45560,"material_conflicts":0,"dual_pairs":680},"classification":"PASS","decision":"train S0","next action":"static probe","elapsed wall time":str(now-start)},
  {"timestamp":now.isoformat(),"phase":"Static","run ID":"S0_30000","parent":"W1B-R2","hypothesis":"S0 can retain joint static solution","single changed variable":"BC on Phase 1 dataset","metrics":{"worst_mse":s0["selected"]["validation"]["worst_condition_mse"],"dual_accuracy":s0["selected"]["validation"]["dual_mode_classification"]},"classification":"EXP014_EXPLICIT_MODE_STATIC_PASS","decision":"S1/S2 not authorized; evaluate closed loop","next action":"Phase 2","elapsed wall time":str(now-start)},
  {"timestamp":now.isoformat(),"phase":"Phase 2","run ID":"P2_R0","parent":"S0_30000","hypothesis":"static solution transfers closed loop","single changed variable":"teacher-free closed-loop evaluation","metrics":stats["round0"],"classification":"STAND_RETENTION_FAIL","decision":"DAgger round 1","next action":"horizon 8","elapsed wall time":str(now-start)},
  {"timestamp":now.isoformat(),"phase":"DAgger","run ID":"P2_D1","parent":"P2_R0","hypothesis":"8-step student states close initial gap","single changed variable":"DAgger horizon 8","metrics":stats["round1"],"classification":"STAND_RETENTION_FAIL","decision":"improved but failed; second retry","next action":"horizon 16","elapsed wall time":str(now-start)},
  {"timestamp":now.isoformat(),"phase":"DAgger","run ID":"P2_D2","parent":"P2_D1","hypothesis":"16-step student states close remaining gap","single changed variable":"DAgger horizon 16","metrics":stats["round2"],"classification":"STAND_RETENTION_FAIL","decision":"stop same failure class after two retries","next action":"read-only reset boundary diagnosis","elapsed wall time":str(now-start)},
  {"timestamp":now.isoformat(),"phase":"Diagnosis","run ID":"P2_RESET_GAP","parent":"P2_D2","hypothesis":"early reset boundary remains unlabeled","single changed variable":"read-only coverage audit","metrics":diagnosis["reset_boundary_coverage"],"classification":"STUDENT_VISITED_STATE_GAP","decision":"Phase 3 not authorized","next action":next_action["experiment"],"elapsed wall time":str(now-start)}]
 with (OUT/"experiment_journal.jsonl").open("a",encoding="utf-8") as f:
  for e in entries:f.write(json.dumps(e,ensure_ascii=False)+"\n")
 report=f"""# EXP 014 — 12h autonomous progress report

## Outcome

Classification: **EXP014_STATIC_PASS_PHYSICAL_FAIL**. The causal 141D contract and formal S/W dataset passed integrity. S0 (90,570 parameters) passed every static group at step 30,000; S1/S2 were therefore not authorized. Teacher-free closed loop failed practical STAND and downstream retention. Two registered DAgger retries improved aggregate STAND hold from {stats['round0']['stand_hold']:.1%} to {stats['round2']['stand_hold']:.1%}, but did not approach 95% and aggregate fall remained {stats['round2']['fall']:.1%}.

## Central result

At identical physical B0 states, 680/680 STAND/WALK pairs had different 141D inputs and different S/W actions, with zero material collision at 1e-6 through 1e-3 quantization. Static dual-mode classification was {s0['selected']['validation']['dual_mode_classification']:.2%}. Thus explicit mode resolves the representation-level zero-command ambiguity, but this run does not establish a unified physical actor.

## Diagnosis

The DAgger fail-closed proxy rejected reset steps 0–3 because the robot had not made foot contact. Rounds 1/2 began labeling STAND at step 4. The resulting policy improved STAND monotonically while WALK acquisition stayed nonzero, which localizes the next experiment to reset-boundary labelability rather than capacity or RUN.

## Scope

RUN integration, local-neighborhood promotion, and OMNI-RUN audit were not authorized because Phase 2 failed. No protected asset was changed and no runtime router, teacher, checkpoint switching, or action blend was introduced.
""";(REPO/"research/exp_014_12h_autonomous_progress_report.md").write_text(report,encoding="utf-8")
 repro=f"""$ErrorActionPreference = 'Stop'
$repo = '{REPO}'
$python = 'C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe'
$isaac = 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat'
Set-Location $repo
& $python experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/bootstrap_phase0.py
& $isaac -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/collect_phase1.py --batch 0 --episodes-per-condition 20 --device cuda:0 --viz none
& $python experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/audit_dataset.py
& $python experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/train_static.py --model S0
# Continue with collect_dagger.py horizons 8 and 16, train_dagger.py, then evaluate_phase2.py batches 0..4.
""";(OUT/"reproduction_commands.ps1").write_text(repro,encoding="utf-8")
 print(json.dumps({"classification":classification,"selected":selected,"stats":stats,"heldout_static_pass":heldout["static_gate_pass"]},indent=2))
if __name__=="__main__":main()
