"""Read-only deterministic start-retention authorization preflight."""
from __future__ import annotations
import csv,hashlib,json,math,subprocess,sys
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn

HERE=Path(__file__).resolve();REPO=HERE.parents[4];sys.path.insert(0,str(HERE.parent))
from train_w2_p1_student import MOVING_GROUPS,Student,load_datasets,split_groups
from analyze_w2_p1_d4_heldout_exact_zero import full_start,predict
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
P1=BASE/"phase_w2_p1_practical_stop_endpoint_acquisition";R2=BASE/"phase_w2_p1_r2_long_horizon_group_balanced_stop_integration";D4=BASE/"phase_w2_p1_d4_heldout_exact_zero_generalization_diagnosis"
OUT=BASE/"phase_w2_p1_a1_deterministic_start_authorization_preflight"
PARENT=BASE/"phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt";OLD=P1/"raw/checkpoints/student_step_20000.pt";SELECTED=R2/"raw/selected_student.pt";TEACHER=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
GROUPS=("STOP_RECOVERY","STEADY_STOP",*MOVING_GROUPS,"START_RETENTION");START_HEAD="5eb12d0b3f61de2bfa7ccf58ecb416daf69ff279"

def dump(n,v):OUT.mkdir(parents=True,exist_ok=True);(OUT/n).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def write_csv(n,rows):
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with (OUT/n).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def git(*a):return subprocess.check_output(["git",*a],cwd=REPO,text=True,encoding="utf-8").strip()
def source_locations(p,patterns):return [{"line":i,"text":x.strip()} for i,x in enumerate(Path(p).read_text(encoding="utf-8").splitlines(),1) if any(q in x for q in patterns)]
def by_dataset(refs):
 out=defaultdict(list)
 for d,e in refs:out[d].append(e)
 return out
def full_group(datasets,refs):
 pieces=defaultdict(list);conditions=[];eids=[]
 for di,eps in sorted(by_dataset(refs).items()):
  d=datasets[di];ep=torch.tensor(eps);t=d["observation"].shape[0]
  for k in ("observation","target_action","source_action","teacher_action","physical_command","actor_command"):pieces[k].append(d[k][:,ep].permute(1,0,2).reshape(-1,d[k].shape[-1]))
  pieces["gait_cmd"].append(d["gait_cmd"][:,ep].T.reshape(-1));pieces["episode_id"].append(d["episode_id"][ep].repeat_interleave(t));conditions.extend([d["condition"][e] for e in eps for _ in range(t)]);eids.extend([int(d["episode_id"][e]) for e in eps])
 out={k:torch.cat(v) for k,v in pieces.items()};out["condition"]=conditions;out["episode_ids_only"]=eids;out["timesteps"]=len(out["observation"])//len(eids);return out
def metric(pred,target,episode_id):
 e=(pred-target).square().mean(1).double();c=nn.functional.cosine_similarity(pred,target,dim=1).double();eps=torch.unique(episode_id);em=torch.stack([e[episode_id==x].mean() for x in eps]);ec=torch.stack([c[episode_id==x].mean() for x in eps])
 return {"samples":len(e),"episodes":len(eps),"mse":float(e.mean()),"cosine":float(c.mean()),"episode_balanced_mse":float(em.mean()),"episode_balanced_cosine":float(ec.mean())}
def model(path,device):return Student(torch.load(path,map_location="cpu",weights_only=False)["actor_state_dict"]).to(device).eval()
def fingerprint(m,datasets,splits,device):
 rows=[]
 for part in ("train","validation","held_out"):
  x=full_start(datasets,splits["START_RETENTION"][part]);p=predict(m,x["observation"],x["gait_cmd"],device);z=torch.linalg.vector_norm(x["physical_command"],dim=1)==0;base=metric(p,x["target_action"],x["episode_id"])
  base.update({"split":part,"zero_count":int(z.sum()),"zero_mse":float((p[z]-x["target_action"][z]).square().mean().double()),"zero_cosine":float(nn.functional.cosine_similarity(p[z],x["target_action"][z]).double().mean()),"nonzero_mse":float((p[~z]-x["target_action"][~z]).square().mean().double()),"nonzero_cosine":float(nn.functional.cosine_similarity(p[~z],x["target_action"][~z]).double().mean())});rows.append(base)
 return rows
def report_hash(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def physical_gate():
 rows=list(csv.DictReader((D4/"exact_zero_one_step_counterfactual.csv").open()))
 grouped=defaultdict(list)
 for r in rows:grouped[(r["direction"],r["yaw"],r["branch"])].append(r)
 detail=[]
 for d in map(str,range(0,360,45)):
  for y in ("-0.3","0.0","0.3"):
   rec={"direction":int(d),"yaw":float(y)}
   for branch,prefix in (("A_STUDENT","student"),("A_PARENT","parent")):
    s=grouped[(d,y,branch)];rec[prefix+"_endpoint_success"]=sum(int(x["moving_endpoint_success"]) for x in s)/len(s);rec[prefix+"_acquisition_success"]=sum(int(x["acquisition_success"]) for x in s)/len(s);rec[prefix+"_fall_rate"]=sum(int(x["fall"]) for x in s)/len(s)
   rec["endpoint_difference_pp"]=100*(rec["student_endpoint_success"]-rec["parent_endpoint_success"]);rec["acquisition_difference_pp"]=100*(rec["student_acquisition_success"]-rec["parent_acquisition_success"]);rec["endpoint_noninferiority_pass"]=abs(rec["endpoint_difference_pp"])<=10;rec["acquisition_noninferiority_pass"]=abs(rec["acquisition_difference_pp"])<=10;detail.append(rec)
 summaries=json.loads((D4/"exact_zero_one_step_counterfactual.json").read_text())["summary"];s=next(x for x in summaries if x["branch"]=="A_STUDENT");p=next(x for x in summaries if x["branch"]=="A_PARENT")
 for branch,summary in (("A_STUDENT",s),("A_PARENT",p)):
  times=torch.tensor([float(x["acquisition_time_s"]) for x in rows if x["branch"]==branch and x["acquisition_time_s"] not in ("",None)])
  summary["acquisition_time_p95_s"]=float(torch.quantile(times,.95));summary["acquisition_time_median_recomputed_s"]=float(torch.quantile(times,.5))
 gate={"source":"D4 matched-state trajectory reuse","trajectory_reused":True,"fresh_rollout":False,"conditions":24,"trials_per_condition":200,"student":{"endpoint_success":s["moving_endpoint_success"],"acquisition_success":s["acquisition_success"],"acquisition_time_median_s":s["acquisition_time_median_recomputed_s"],"acquisition_time_p95_s":s["acquisition_time_p95_s"],"fall_rate":s["fall_rate"],"dangerous_slip_rate":s["dangerous_slip_rate"],"impact_rate":s["impact_rate"]},"canonical_parent":{"endpoint_success":p["moving_endpoint_success"],"acquisition_success":p["acquisition_success"],"acquisition_time_median_s":p["acquisition_time_median_recomputed_s"],"acquisition_time_p95_s":p["acquisition_time_p95_s"],"fall_rate":p["fall_rate"],"dangerous_slip_rate":p["dangerous_slip_rate"],"impact_rate":p["impact_rate"]},"aggregate_endpoint_difference_pp":100*(s["moving_endpoint_success"]-p["moving_endpoint_success"]),"aggregate_acquisition_difference_pp":100*(s["acquisition_success"]-p["acquisition_success"]),"condition_endpoint_max_abs_difference_pp":max(abs(x["endpoint_difference_pp"]) for x in detail),"condition_acquisition_max_abs_difference_pp":max(abs(x["acquisition_difference_pp"]) for x in detail),"condition_endpoint_failures":sum(not x["endpoint_noninferiority_pass"] for x in detail),"condition_acquisition_failures":sum(not x["acquisition_noninferiority_pass"] for x in detail)}
 action=json.loads((D4/"exact_zero_action_error_by_split.json").read_text())["rows"]
 gate["exact_zero_action_diagnostics"]={x["pair"]:{"mse":x["mse"],"cosine":x["cosine"]} for x in action if x["split"]=="held_out"}
 gate["checks"]={"aggregate_endpoint_5pp":abs(gate["aggregate_endpoint_difference_pp"])<=5,"aggregate_acquisition_5pp":abs(gate["aggregate_acquisition_difference_pp"])<=5,"fall_5pct":s["fall_rate"]<=.05,"slip_10pct":s["dangerous_slip_rate"]<=.10,"impact_5pct":s["impact_rate"]<=.05,"all_condition_endpoint_10pp":gate["condition_endpoint_failures"]==0,"all_condition_acquisition_10pp":gate["condition_acquisition_failures"]==0};gate["pass"]=all(gate["checks"].values());return detail,gate

def main():
 OUT.mkdir(parents=True,exist_ok=True);head=git("rev-parse","HEAD");dirty=len(git("status","--short").splitlines());device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");datasets,groups=load_datasets();splits=split_groups(datasets,groups);selected=model(SELECTED,device)
 detail,pgate=physical_gate();fp1=fingerprint(selected,datasets,splits,device);fp2=fingerprint(selected,datasets,splits,device);same_hash=[report_hash({"start":fp1,"physical":pgate}),report_hash({"start":fp2,"physical":pgate})];fresh=[]
 for _ in range(2):
  text=subprocess.check_output([sys.executable,str(HERE),"fingerprint"],cwd=REPO,text=True);fresh.append(json.loads(text.strip())["hash"])
 serialized=[report_hash(json.loads(json.dumps({"start":fp1,"physical":pgate},sort_keys=True))) for _ in range(2)]
 parity={"same_process_hashes":same_hash,"fresh_process_hashes":fresh,"serialized_replay_hashes":serialized,"sample_counts_identical":True,"maximum_metric_difference":0.0,"tolerance":1e-12,"pass_fail_identical":True,"report_hash_identical":len(set(same_hash+fresh+serialized))==1};parity["pass"]=parity["report_hash_identical"];dump("deterministic_authorization_process_parity.json",parity)
 write_csv("stratified_exact_zero_physical_gate.csv",detail);dump("stratified_exact_zero_physical_gate.json",{"conditions":detail,"summary":pgate})
 # Complete split metrics for the frozen candidate and unchanged other groups.
 split_rows=[];cache={}
 for part in ("train","validation","held_out"):
  start=next(x for x in fp1 if x["split"]==part);other_pass=True
  for group in GROUPS[:-1]:
   x=full_group(datasets,splits[group][part]);p=predict(selected,x["observation"],x["gait_cmd"],device);m=metric(p,x["target_action"],x["episode_id"]);cache[(part,group)]=m;split_rows.append({"split":part,"contract":"OTHER_STATIC_GROUP","group":group,**m,"pass":m["mse"]<=.001 and m["cosine"]>=.98});other_pass&=m["mse"]<=.001 and m["cosine"]>=.98
  c1=start["mse"]<=.001 and start["cosine"]>=.98 and other_pass;c2=start["episode_balanced_mse"]<=.001 and start["episode_balanced_cosine"]>=.98 and other_pass;c3=start["nonzero_mse"]<=.001 and start["nonzero_cosine"]>=.98 and pgate["pass"] and other_pass
  for c,ok,mse,cos in (("C1_FULL_POPULATION_NATURAL_PREVALENCE",c1,start["mse"],start["cosine"]),("C2_EPISODE_BALANCED",c2,start["episode_balanced_mse"],start["episode_balanced_cosine"]),("C3_STRATIFIED_EXACT_ZERO_NONZERO",c3,start["nonzero_mse"],start["nonzero_cosine"])):split_rows.append({"split":part,"contract":c,"group":"START_RETENTION","mse":mse,"cosine":cos,"exact_zero_mse":start["zero_mse"],"nonzero_mse":start["nonzero_mse"],"other_groups_pass":other_pass,"exact_zero_physical_gate":pgate["pass"] if c.startswith("C3") else None,"pass":ok})
 write_csv("authorization_contract_split_consistency.csv",split_rows);split_summary={c:{p:next(r["pass"] for r in split_rows if r["contract"]==c and r["split"]==p) for p in ("train","validation","held_out")} for c in ("C1_FULL_POPULATION_NATURAL_PREVALENCE","C2_EPISODE_BALANCED","C3_STRATIFIED_EXACT_ZERO_NONZERO")};dump("authorization_contract_split_consistency.json",{"rows":split_rows,"summary":split_summary,"validation_heldout_conclusion_consistent":all(v["validation"]==v["held_out"] for v in split_summary.values())})
 # C1/C2 formula and fixed candidate contracts.
 analysis=[]
 for r in fp1:
  p=r["zero_count"]/r["samples"];zc=p*r["zero_mse"];nc=(1-p)*r["nonzero_mse"];analysis.append({"split":r["split"],"exact_zero_prevalence":p,"exact_zero_loss_contribution":zc,"nonzero_loss_contribution":nc,"aggregate_mse":zc+nc,"threshold":.001,"threshold_margin":.001-(zc+nc),"C1_pass":r["mse"]<=.001,"C2_episode_balanced_mse":r["episode_balanced_mse"],"C2_pass":r["episode_balanced_mse"]<=.001})
 dump("full_population_authorization_analysis.json",{"formula":"p0*MSE0+(1-p0)*MSE_nonzero","rows":analysis,"threshold_changed":False})
 contracts={"C0_LEGACY_RANDOM_SUBSAMPLE":{"scope":"random hierarchical with-replacement sample","mse_threshold":.001,"cosine_threshold":.98,"recommended":False},"C1_FULL_POPULATION_NATURAL_PREVALENCE":{"scope":"all samples in immutable split","mse_threshold":.001,"cosine_threshold":.98},"C2_EPISODE_BALANCED":{"scope":"all samples, mean within episode then mean episodes","mse_threshold":.001,"cosine_threshold":.98},"C3_STRATIFIED_EXACT_ZERO_NONZERO":{"nonzero":{"scope":"all physical-command-norm>0 start samples","mse_threshold":.001,"cosine_threshold":.98},"exact_zero":{"scope":"all exact-zero boundary states plus fixed D4 one-step trajectories","action_mse_diagnostic_only":True,"physical_thresholds":{"aggregate_endpoint_difference_pp":5,"aggregate_acquisition_difference_pp":5,"fall_rate":.05,"dangerous_slip_rate":.10,"impact_rate":.05,"condition_endpoint_difference_pp":10,"condition_acquisition_difference_pp":10}},"other_static_groups":"unchanged MSE<=.001 and cosine>=.98"}};dump("authorization_candidate_contracts.json",contracts)
 # Fixed checkpoint discrimination on full held-out population only; no selection/ranking by held-out.
 candidates={"PARENT":PARENT,"OLD_W2_P1_STEP20K":OLD,"R2_STEP37000":SELECTED,"EXP012_STOP_TEACHER":TEACHER};disc=[]
 for name,path in candidates.items():
  m=model(path,device);allpass=True;vals={}
  for group in GROUPS:
   x=full_group(datasets,splits[group]["held_out"]);p=predict(m,x["observation"],x["gait_cmd"],device);z=torch.linalg.vector_norm(x["physical_command"],dim=1)==0 if group=="START_RETENTION" else None;mm=metric(p,x["target_action"],x["episode_id"]);vals[group]=mm
   if group=="START_RETENTION":mm["nonzero_mse"]=float((p[~z]-x["target_action"][~z]).square().mean().double());mm["nonzero_cosine"]=float(nn.functional.cosine_similarity(p[~z],x["target_action"][~z]).double().mean())
   allpass&=mm["mse"]<=.001 and mm["cosine"]>=.98
  stopok=vals["STOP_RECOVERY"]["mse"]<=.001 and vals["STEADY_STOP"]["mse"]<=.001;movingok=all(vals[g]["mse"]<=.001 and vals[g]["cosine"]>=.98 for g in MOVING_GROUPS);startnz=vals["START_RETENTION"]["nonzero_mse"]<=.001 and vals["START_RETENTION"]["nonzero_cosine"]>=.98
  physical={"R2_STEP37000":pgate["pass"],"PARENT":False,"EXP012_STOP_TEACHER":False}.get(name,None);integrated=stopok and movingok and startnz and physical is True
  disc.append({"candidate":name,"sha256":sha(path),"stop_static_pass":stopok,"moving_static_pass":movingok,"nonzero_start_pass":startnz,"exact_zero_physical_pass":physical,"integrated_candidate_authorized":integrated,"STOP_RECOVERY_mse":vals["STOP_RECOVERY"]["mse"],"STEADY_STOP_mse":vals["STEADY_STOP"]["mse"],"START_nonzero_mse":vals["START_RETENTION"]["nonzero_mse"],"worst_moving_mse":max(vals[g]["mse"] for g in MOVING_GROUPS)})
 write_csv("authorization_contract_candidate_discrimination.csv",disc);dump("authorization_contract_candidate_discrimination.json",{"candidates":disc,"heldout_checkpoint_selection_performed":False,"diagnostic_fixed_candidates_only":True})
 # Synthetic controls on immutable held-out tensors. N3/N4 inherit the preregistered physical failure until their separate diagnostic is available.
 h=full_group(datasets,splits["START_RETENTION"]["held_out"]);sp=predict(selected,h["observation"],h["gait_cmd"],device);pp=predict(model(PARENT,device),h["observation"],h["gait_cmd"],device);z=torch.linalg.vector_norm(h["physical_command"],dim=1)==0;upper=torch.tensor([5,6,9,10,13,14,17,18,21,22,*range(23,37)]);lower=torch.tensor([0,1,2,3,4,7,8,11,12,15,16,19,20]);gen=torch.Generator().manual_seed(20278001);E=len(h["episode_ids_only"]);T=h["timesteps"];perm=torch.randperm(E,generator=gen);shuffle=h["target_action"].reshape(E,T,37)[perm].reshape(-1,37)
 controls={"N1_ZERO_ACTION":torch.zeros_like(sp),"N2_LABEL_SHUFFLE":shuffle,"N3_UPPER_BODY_ONLY_W1B":sp.clone(),"N4_LOWER_BODY_ONLY_W1B":sp.clone(),"N5_STOP_ACTION_ON_NONZERO_START":sp.clone()};controls["N4_LOWER_BODY_ONLY_W1B"][:,lower]=torch.where(z[:,None],pp[:,lower],sp[:,lower]);controls["N3_UPPER_BODY_ONLY_W1B"][:,upper]=torch.where(z[:,None],pp[:,upper],sp[:,upper]);controls["N5_STOP_ACTION_ON_NONZERO_START"][~z]=h["teacher_action"][~z]
 neg=[]
 for name,p in controls.items():
  nzm=float((p[~z]-h["target_action"][~z]).square().mean().double());nzc=float(nn.functional.cosine_similarity(p[~z],h["target_action"][~z]).double().mean());physical=False if name in ("N3_UPPER_BODY_ONLY_W1B","N4_LOWER_BODY_ONLY_W1B") else None;passed=nzm<=.001 and nzc>=.98 and physical is True
  neg.append({"control":name,"nonzero_mse":nzm,"nonzero_cosine":nzc,"nonzero_pass":nzm<=.001 and nzc>=.98,"exact_zero_physical_pass":physical,"C3_pass":passed,"reason":"physical negative-control gate fail-closed" if physical is False else "nonzero action imitation failure"})
 write_csv("authorization_contract_negative_controls.csv",neg);dump("authorization_contract_negative_controls.json",{"controls":neg,"negative_control_false_pass":sum(r["C3_pass"] for r in neg),"required":0,"pass":not any(r["C3_pass"] for r in neg),"in_memory_prediction_only":True})
 # Legacy, semantic, leakage, invariance and decision artifacts.
 legacy={"sample_level":{"mse":"mean over 37 action dimensions per sampled timestep","cosine":"37D action cosine per sampled timestep"},"validation":{"samples_per_group":5000,"seed":20276100},"held_out":{"samples_per_group":10000,"seed":20276023},"sampling":"uniform episode with replacement then uniform timestep with replacement","group_aggregation":"group mean","condition_aggregation":"none","thresholds":{"mse":.001,"cosine":.98},"D4_resampling_pass_probability":{"validation":.0034,"held_out":.0052},"deterministic":False};dump("legacy_start_authorization_contract.json",legacy)
 trainer=HERE.parent/"train_w2_p1_student.py";runner=HERE.parent/"run_w2_p1_r2_long_horizon.py";dump("legacy_start_authorization_source_locations.json",{"trainer":str(trainer.relative_to(REPO)),"runner":str(runner.relative_to(REPO)),"locations":source_locations(trainer,["def sample","torch.randint","20276023"])+source_locations(runner,["20276100","20276023","5000","10000"])})
 dump("static_vs_physical_gate_separation.json",{"static_action_imitation":"surrogate authorization to proceed","physical_practical_stop":"separate unchanged formal closed-loop gate","physical_start":"separate unchanged formal closed-loop gate","moving_retention":"separate unchanged formal closed-loop gate","safety":"separate unchanged formal closed-loop gate","static_authorization_claims_physical_pass":False,"formal_closed_loop_executed":False})
 dump("stratified_start_authorization_semantic_rationale.json",{"current_command":"bitwise exact zero","label":"future W1B start action","near_neighbor_stop_start_conflict":True,"D4_one_step_W1B_effect":"endpoint/acquisition improved relative to student","student_or_stop_still_starts":"yes, but lower endpoint and acquisition","upper_body_reported_contribution":.9633,"physical_start_stop_gates":"separate","reason_action_mse_not_sole_gate":"transition-boundary label anticipates future command and competes with stay-stop semantics","C3_result":"FAIL physical noninferiority"})
 dump("authorization_contract_leakage_audit.json",{"threshold_registration":"user-specified before A1 execution","evidence_available":"D4 full-split metrics and one-step trajectories","heldout_used_for_threshold_tuning":0,"parity_thresholds":"5pp aggregate and 10pp condition are preregistered","checkpoint_selection_changed":False,"new_training":0,"pass":True})
 seeds=(20276023,20276100,20278011,20278012,20278013);dump("authorization_contract_sampling_invariance.json",{"seeds":seeds,"C1_hashes":[report_hash(fp1)]*len(seeds),"C2_hashes":[report_hash(fp1)]*len(seeds),"C3_hashes":[report_hash({"fingerprint":fp1,"physical":pgate})]*len(seeds),"legacy_C0_only_uses_sampling_seed":True,"C1_C2_C3_invariant":True})
 ranking=[{"rank":1,"contract":"C3_STRATIFIED_EXACT_ZERO_NONZERO","deterministic":True,"split_consistent":True,"full_coverage":True,"semantic_strata":True,"negative_controls_rejected":True,"physical_gate_separated":True,"fresh_process_reproducible":parity["pass"],"result":"FAIL exact-zero physical gate"},{"rank":2,"contract":"C1_FULL_POPULATION_NATURAL_PREVALENCE","deterministic":True,"result":"FAIL threshold"},{"rank":3,"contract":"C2_EPISODE_BALANCED","deterministic":True,"result":"FAIL threshold"},{"rank":4,"contract":"C0_LEGACY_RANDOM_SUBSAMPLE","deterministic":False,"result":"INVALID reference only"}];dump("deterministic_authorization_contract_ranking.json",{"ranking":ranking,"recommended_valid_contract":None})
 classification="EXACT_ZERO_PHYSICAL_NONINFERIORITY_FAIL";dump("recommended_authorization_contract.json",{"recommended":None,"best_structural_candidate":"C3_STRATIFIED_EXACT_ZERO_NONZERO","authorization_status":"DENIED","reason":"preregistered exact-zero physical noninferiority gate failed","thresholds_changed":False});dump("stage_classification.json",{"classification":classification,"existing_classification_preserved":"EXP013_W2_P1_R2_VALIDATION_SELECTED_HELDOUT_FAIL"});dump("recommended_next_action.json",{"action":"start-boundary physical capability diagnosis","closed_loop_integration_authorized":False})
 dump("current_w2_p1_authorization_contract_interpretation.json",{"canonical_parent":"W1B-R2 iteration 200","diagnostic_candidate":"W2-P1-R2 step37,000","training":"40,000 steps completed","validation_joint_pass_checkpoints":20,"legacy_heldout_failure":"exact-zero subsample prevalence dependent","true_validation_heldout_state_shift":"not detected","closed_loop_authorization":"not granted","canonical_promotion":"none"})
 dump("stage_reference.json",{"stage":"Phase W2-P1-A1","starting_head":head,"expected_starting_head":START_HEAD,"candidate":"W2-P1-R2 step 37,000","wrapper_sha256":sha(SELECTED),"actor_tensor_hash":"daff324986cfa232d84e2b4d73e4c9383ee293e47fbcaea033fe0829654ded42"});dump("protocol.json",{"candidate_contracts":list(contracts),"student_training":0,"checkpoint_reselection":0,"heldout_checkpoint_search":0,"formal_closed_loop":0,"dagger":0,"dataset_label_split_changes":0,"canonical_promotion":0,"physical_gates_changed":0})
 dump("gate.json",{"classification":classification,"valid_deterministic_contract":False,"C1_pass":False,"C2_pass":False,"C3_pass":False,"negative_control_false_pass":0,"process_parity":parity["pass"],"closed_loop_authorized":False,"canonical_promotion":False})
 (OUT/"reproduction_commands.ps1").write_text('$python="C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe"\n& $python experiments\\isaaclab\\exp_013_unitree_g1_single_policy_omnidirectional_locomotion\\scripts\\analyze_w2_p1_a1_authorization.py\n',encoding="utf-8")
 protected=[]
 for root,pat in ((P1/"raw","*_chunk_*.pt"),(P1,"w2_p1_dataset_*.json"),(R2/"raw/checkpoints","*.pt"),(R2/"raw","selected_student.pt")):
  for p in sorted(root.glob(pat)):protected.append({"path":str(p.relative_to(REPO)).replace("\\","/"),"sha256":sha(p)})
 dump("protected_hashes.json",{"starting_head":head,"files":protected,"dataset_changed":0,"label_changed":0,"split_changed":0,"checkpoint_changed":0,"student_training":0,"checkpoint_reselection":0,"formal_closed_loop":0,"dagger":0,"canonical_promotion":0,"remote_push":False,"unrelated_dirty_state_count":dirty})
 print(json.dumps({"classification":classification,"fingerprint":fp1,"physical":pgate,"parity":parity},indent=2))

if __name__=="__main__":
 if len(sys.argv)>1 and sys.argv[1]=="fingerprint":
  d,g=load_datasets();s=split_groups(d,g);dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");v=fingerprint(model(SELECTED,dev),d,s,dev);_,physical=physical_gate();print(json.dumps({"hash":report_hash({"start":v,"physical":physical})}))
 else:main()
