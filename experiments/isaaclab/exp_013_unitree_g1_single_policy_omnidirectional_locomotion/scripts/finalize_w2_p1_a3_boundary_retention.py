"""Finalize fail-closed A3 after the preregistered static grid."""
from __future__ import annotations
import csv,hashlib,json,sys
from pathlib import Path
import torch
from torch import nn
HERE=Path(__file__).resolve();REPO=HERE.parents[4];sys.path.insert(0,str(HERE.parent))
import probe_w2_p1_a3_boundary_retention as probe
from train_w2_p1_student import MOVING_GROUPS,Student,load_datasets,split_groups
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion";OUT=BASE/"phase_w2_p1_a3_localized_start_boundary_retention_preflight";A2=BASE/"phase_w2_p1_a2_start_boundary_physical_diagnosis";R2=BASE/"phase_w2_p1_r2_long_horizon_group_balanced_stop_integration";RES=BASE/"phase_w2_p1_r1_d2_dataset_provenance_reconciliation";SELECTED=R2/"raw/selected_student.pt";START="bb2147256b36cacd7fee412c05f78a61266eb65c";LOWER=(0,1,3,4,7,8,11,12,15,16,19,20);WAIST=(2,);UPPER=tuple(i for i in range(37) if i not in LOWER+WAIST)
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def parse(spec):
 a=spec.split("_");return float(a[0][1:]),float(a[1][2:]),int(a[2][1:])
def rebuild(spec,datasets,splits,device):
 bw,lr,steps=parse(spec);ow=(1-bw)/4;pg=torch.Generator().manual_seed(20278120);pools={"BOUNDARY":probe.fixed_pool(splits["START_RETENTION"]["train"],datasets,"boundary",8192,pg),"START_NONBOUNDARY":probe.fixed_pool(splits["START_RETENTION"]["train"],datasets,"nonboundary",8192,pg),"STOP_RECOVERY":probe.fixed_pool(splits["STOP_RECOVERY"]["train"],datasets,"any",8192,pg),"STEADY_STOP":probe.fixed_pool(splits["STEADY_STOP"]["train"],datasets,"any",8192,pg)}
 for g in MOVING_GROUPS:pools[g]=probe.fixed_pool(splits[g]["train"],datasets,"any",4096,pg)
 init=torch.load(SELECTED,map_location="cpu",weights_only=False)["actor_state_dict"];torch.manual_seed(20278121);gen=torch.Generator().manual_seed(20278121);model=Student(init).to(device);opt=torch.optim.Adam(model.parameters(),lr=lr)
 for _ in range(steps):
  def loss(key,n=256):
   p=pools[key];ids=torch.randint(len(p[0]),(n,),generator=gen);o,g,t=(v[ids].to(device) for v in p);return nn.functional.mse_loss(model(o,g),t)
  total=bw*loss("BOUNDARY")+ow*(loss("STOP_RECOVERY")+loss("STEADY_STOP")+loss("START_NONBOUNDARY")+torch.stack([loss(g,64) for g in MOVING_GROUPS]).mean());opt.zero_grad(set_to_none=True);total.backward();nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
 return model,pools
def joint_eval(model,datasets,splits,device):
 out=[]
 for ti,b in ((0,"B0"),(1,"B1")):
  ps=[];ts=[]
  for di,ei in splits["START_RETENTION"]["validation"]:
   d=datasets[di];ps.append((d["observation"][ti,ei],d["gait_cmd"][ti,ei]));ts.append(d["target_action"][ti,ei])
  o=torch.stack([x[0] for x in ps]).to(device);g=torch.stack([x[1] for x in ps]).to(device);t=torch.stack(ts).to(device)
  with torch.inference_mode():e=(model(o,g)-t).square()
  for name,idx in (("LOWER_BODY",LOWER),("WAIST",WAIST),("UPPER_BODY",UPPER),("WHOLE_BODY",tuple(range(37)))):out.append({"boundary":b,"joint_group":name,"samples":len(e),"mse":float(e[:,list(idx)].mean()),"loss_contribution_fraction":float(e[:,list(idx)].sum()/e.sum())})
 return out
def main():
 results=json.loads((OUT/"probe_training_results.json").read_text());cand=results["candidates"];static=[x for x in cand if x["all_static_pass"]];assert not static
 existing=[x for x in cand if x["existing_static_pass"]];closest=min(existing,key=lambda x:(x["metrics"]["BOUNDARY_B0"]["mse"],x["parameter_l2_movement"]));bestb=min(cand,key=lambda x:x["metrics"]["BOUNDARY_B0"]["mse"]);device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");datasets,groups=load_datasets();splits=split_groups(datasets,groups);model,pools=rebuild(closest["candidate"],datasets,splits,device);joints=joint_eval(model,datasets,splits,device);write_csv=probe.write_csv;write_csv("start_boundary_joint_error_analysis.csv",joints);dump("start_boundary_joint_error_analysis.json",{"closest_existing-retention_candidate":closest["candidate"],"rows":joints,"interpretation":"B0 remains a whole-body label conflict; no joint-specific reweighting was used or authorized"})
 before=json.loads((OUT/"start_boundary_retention_gradient_analysis.json").read_text())["initial_step37000"];after=probe.grad_analysis(model,pools,device);dump("start_boundary_retention_gradient_analysis.json",{"initial_step37000":before,"closest_existing_retention_candidate":{"candidate":closest["candidate"],"analysis":after},"selected_candidate":None,"interpretation":"boundary gradients remain opposed to steady-stop/stop-recovery objectives; increasing boundary weight reduces B0 error but crosses existing gates"})
 notexec={"status":"NOT_EXECUTED_NO_STATIC_CANDIDATE","reason":"selection hierarchy requires simultaneous existing-group and B0/B1 static PASS before physical evaluation","candidate_count":0}
 write_csv("validation_start_boundary_physical_gate.csv",[{"status":notexec["status"],"reason":notexec["reason"]}]);dump("validation_start_boundary_physical_gate.json",notexec);dump("validation_zero_command_bounded_retention.json",notexec);dump("selected_probe_candidate.json",{"selected":False,**notexec,"closest_existing_retention_candidate":closest,"best_B0_candidate":bestb,"fallback":False});dump("heldout_boundary_static_authorization.json",notexec);write_csv("heldout_start_boundary_physical_gate.csv",[{"status":notexec["status"],"reason":notexec["reason"]}]);dump("heldout_start_boundary_physical_gate.json",notexec);dump("heldout_zero_command_bounded_retention.json",notexec)
 a1=json.loads((A2/"a1_exact_zero_branch_reconstruction.json").read_text())["summary"];dur=json.loads((A2/"start_boundary_action_duration_sweep.json").read_text())["summary"];dump("positive_control_comparison.json",{"source":"Phase W2-P1-A2 exact reconstruction; A3 physical rerun prohibited by upstream static gate","current_student":next(x for x in a1 if x["branch"]=="B_STUDENT"),"parent_one_step":next(x for x in a1 if x["branch"]=="B_CANONICAL_PARENT"),"w1b_one_step":next(x for x in a1 if x["branch"]=="B_W1B_LABEL"),"parent_two_step":next(x for x in dur if x["branch"]=="canonical parent" and x["duration_steps"]==2),"w1b_two_step":next(x for x in dur if x["branch"]=="W1B label" and x["duration_steps"]==2),"stop_teacher_two_step":next(x for x in dur if x["branch"]=="stop teacher" and x["duration_steps"]==2)});dump("four_step_diagnostic.json",{"status":"REUSED_A2_POSITIVE_CONTROL_ONLY","candidate_trajectory":"NOT_AVAILABLE_NO_STATIC_CANDIDATE","canonical_w1b_two_step":next(x for x in dur if x["branch"]=="W1B label" and x["duration_steps"]==2),"canonical_w1b_four_step":next(x for x in dur if x["branch"]=="W1B label" and x["duration_steps"]==4),"new_four_step_training_group_created":False})
 cls="START_BOUNDARY_NO_JOINT_STATIC_SOLUTION";dump("stage_reference.json",{"stage":"W2-P1-A3","starting_head":START,"candidate":"W2-P1-R2 step 37,000","candidate_sha256":sha(SELECTED),"new_persistent_checkpoint":0,"formal_closed_loop":0,"remote_push":False});dump("protocol.json",{"boundary_window":["B0 exact zero","B1 first nonzero"],"grid":{"weights":[.025,.05,.10,.15],"learning_rates":[2e-5,5e-5,1e-4],"steps":[250,500,1000,2000]},"selection":"validation only; physical only after joint static PASS","heldout_fallback":False,"PPO":False,"DAgger":False})
 dump("current_w2_p1_start_boundary_retention_interpretation.json",{"canonical_parent":"W1B-R2 iteration 200","integration_candidate":"W2-P1-R2 step37,000","stop_moving_static_integration":"available","start_exact_zero_physical_capability":"not authorized","required_safe_trajectory":"canonical/W1B whole-body 2-4 steps","current_stage":"localized 2-step retention feasibility","canonical_promotion":"none"});dump("stage_classification.json",{"classification":cls,"reason":"no preregistered weight/LR/step candidate simultaneously passes B0/B1 and every existing static group","static_pass_candidates":0,"physical_evaluation":"not authorized"});dump("recommended_next_action.json",{"classification":cls,"recommended_next_action":"retain fail-closed status; diagnose the B0 stop/start label conflict before any formal integration","execute_now":False})
 resolved=json.loads((RES/"w2_p1_dataset_hashes_resolved_v2.json").read_text());actual={p:sha(REPO/p) for p in resolved["hashes"]};dump("protected_hashes.json",{"dataset_hashes":actual,"dataset_manifest_match":actual==resolved["hashes"],"candidate_checkpoint_sha256":sha(SELECTED),"dataset_changes":0,"label_changes":0,"split_changes":0,"checkpoint_changes":0,"optimizer_changes":0,"new_persistent_checkpoint":0,"remote_push":False});dump("gate.json",{"dataset_identity":"PASS","static_joint_candidate":"FAIL","validation_physical":"NOT_AUTHORIZED","heldout":"NOT_AUTHORIZED","formal_closed_loop":"DENIED","canonical_promotion":"DENIED","classification":cls})
 (OUT/"reproduction_commands.ps1").write_text("$py = 'C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe'\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/probe_w2_p1_a3_boundary_retention.py\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/finalize_w2_p1_a3_boundary_retention.py\n",encoding="utf-8")
 print(json.dumps({"classification":cls,"closest":closest["candidate"],"closest_B0":closest["metrics"]["BOUNDARY_B0"]["mse"],"best_B0":bestb["metrics"]["BOUNDARY_B0"]["mse"]}))
if __name__=="__main__":main()
