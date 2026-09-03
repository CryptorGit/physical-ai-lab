"""In-memory A3 localized two-step boundary-retention training grid."""
from __future__ import annotations
import csv,hashlib,json,math,sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
HERE=Path(__file__).resolve();REPO=HERE.parents[4];sys.path.insert(0,str(HERE.parent))
from train_w2_p1_student import MOVING_GROUPS,Student,load_datasets,split_groups
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion";P1=BASE/"phase_w2_p1_practical_stop_endpoint_acquisition";RES=BASE/"phase_w2_p1_r1_d2_dataset_provenance_reconciliation";R2=BASE/"phase_w2_p1_r2_long_horizon_group_balanced_stop_integration";OUT=BASE/"phase_w2_p1_a3_localized_start_boundary_retention_preflight";SELECTED=R2/"raw/selected_student.pt";SPLIT=P1/"w2_p1_dataset_split.json";WEIGHTS=(.025,.05,.10,.15);LRS=(2e-5,5e-5,1e-4);STEPS=(250,500,1000,2000);OTHER=("STOP_RECOVERY","STEADY_STOP","MOVING_RETENTION","START_NONBOUNDARY")
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def hten(x):return sha_bytes(x.detach().cpu().contiguous().numpy().tobytes())
def dump(n,x):OUT.mkdir(parents=True,exist_ok=True);(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def write_csv(n,rows):
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with (OUT/n).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def get_tensor(data,ei,t):return data["observation"][t,ei],data["gait_cmd"][t,ei],data["target_action"][t,ei]
def make_index(datasets,splits):
 rows=[];non=[]
 for part in ("train","validation","held_out"):
  for di,ei in splits["START_RETENTION"][part]:
   d=datasets[di];ep=int(d["episode_id"][ei]);cond=str(d["condition"][ei]);cid=int(d["condition_id"][ei]);final=d["physical_command"][-1,ei];direction=int(round(math.degrees(math.atan2(float(final[1]),float(final[0])))))%360;yaw=float(final[2]);
   for t,bn in ((0,"B0"),(1,"B1")):
    pc=d["physical_command"][t,ei];ac=d["actor_command"][t,ei];rows.append({"split":part,"dataset_index":di,"episode_index":ei,"episode_id":ep,"condition_id":cid,"condition":cond,"direction":direction,"yaw":yaw,"boundary_index":bn,"time_index":t,"physical_command":[float(v) for v in pc],"actor_command":[float(v) for v in ac],"ramp_progress":0.0 if t==0 else 1/75,"observation_sha256":hten(d["observation"][t,ei]),"label_sha256":hten(d["target_action"][t,ei])})
   non.append({"split":part,"dataset_index":di,"episode_index":ei,"episode_id":ep,"time_indices":[2,int(d["observation"].shape[0])-1]})
 return rows,non
def fixed_pool(refs,datasets,kind,count,g):
 obs=torch.empty(count,123);gait=torch.empty(count);target=torch.empty(count,37);choices=torch.randint(len(refs),(count,),generator=g)
 for row,c in enumerate(choices.tolist()):
  di,ei=refs[c];d=datasets[di]
  if kind=="boundary":t=int(torch.randint(2,(1,),generator=g))
  elif kind=="nonboundary":t=int(torch.randint(2,d["observation"].shape[0],(1,),generator=g))
  else:t=int(torch.randint(d["observation"].shape[0],(1,),generator=g))
  obs[row],gait[row],target[row]=get_tensor(d,ei,t)
 return obs,gait,target
def full_eval(model,datasets,splits,part,device):
 groups=("BOUNDARY_B0","BOUNDARY_B1","START_NONBOUNDARY","STOP_RECOVERY","STEADY_STOP",*MOVING_GROUPS);out={};model.eval()
 for group in groups:
  base="START_RETENTION" if group.startswith("BOUNDARY") or group=="START_NONBOUNDARY" else group;refs=splits[base][part];sum_m=sum_c=0.;n=0;sum_l2=0.
  for start in range(0,len(refs),128):
   batch=refs[start:start+128];os=[];gs=[];ts=[]
   for di,ei in batch:
    d=datasets[di]
    times=[0] if group=="BOUNDARY_B0" else [1] if group=="BOUNDARY_B1" else range(2,d["observation"].shape[0]) if group=="START_NONBOUNDARY" else range(d["observation"].shape[0])
    for t in times:o,g,tg=get_tensor(d,ei,t);os.append(o);gs.append(g);ts.append(tg)
   if not os:continue
   o=torch.stack(os).to(device);g=torch.stack(gs).to(device);tg=torch.stack(ts).to(device)
   with torch.inference_mode():p=model(o,g);m=(p-tg).square().mean(1);c=nn.functional.cosine_similarity(p,tg);l=torch.linalg.vector_norm(p-tg,dim=1)
   sum_m+=float(m.sum());sum_c+=float(c.sum());sum_l2+=float(l.sum());n+=len(m)
  out[group]={"samples":n,"mse":sum_m/n,"cosine":sum_c/n,"whole_body_l2":sum_l2/n,"pass":sum_m/n<=.001 and sum_c/n>=.98}
 out["BOUNDARY_COMBINED"]={"mse":(out["BOUNDARY_B0"]["mse"]+out["BOUNDARY_B1"]["mse"])/2,"cosine":(out["BOUNDARY_B0"]["cosine"]+out["BOUNDARY_B1"]["cosine"])/2,"pass":out["BOUNDARY_B0"]["pass"] and out["BOUNDARY_B1"]["pass"]};return out
def grad_analysis(model,pools,device):
 names=("BOUNDARY","STEADY_STOP","STOP_RECOVERY","MOVING_RETENTION","START_NONBOUNDARY");vec={};layer={};joint={}
 for name in names:
  model.zero_grad(set_to_none=True)
  if name=="MOVING_RETENTION":loss=torch.stack([nn.functional.mse_loss(model(pools[g][0].to(device),pools[g][1].to(device)),pools[g][2].to(device)) for g in MOVING_GROUPS]).mean()
  else:
   key="BOUNDARY" if name=="BOUNDARY" else name;o,g,t=pools[key];loss=nn.functional.mse_loss(model(o.to(device),g.to(device)),t.to(device))
  loss.backward();parts=[];layer[name]={}
  for pn,p in model.named_parameters():
   q=torch.zeros_like(p).flatten() if p.grad is None else p.grad.detach().flatten();parts.append(q);layer[name][pn]=float(torch.linalg.vector_norm(q))
  vec[name]=torch.cat(parts);last=model.hidden[-1].weight.grad.detach();joint[name]=[float(torch.linalg.vector_norm(last[j])) for j in range(37)]
 cos={}
 for other in names[1:]:cos[f"BOUNDARY_vs_{other}"]=float(nn.functional.cosine_similarity(vec["BOUNDARY"],vec[other],dim=0))
 return {"pairwise_cosine":cos,"gradient_norms":{k:float(torch.linalg.vector_norm(v)) for k,v in vec.items()},"layerwise_norms":layer,"joint_head_norms":joint}
def main():
 OUT.mkdir(parents=True,exist_ok=True);device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");datasets,groups=load_datasets();splits=split_groups(datasets,groups);idx,non=make_index(datasets,splits);dump("start_boundary_2step_index.json",{"view_only":True,"source_tensors_resaved":False,"entries":idx,"counts":dict((p,sum(r["split"]==p for r in idx)) for p in ("train","validation","held_out")),"B0_contract":"time index 0, bitwise-zero physical/actor command","B1_contract":"time index 1, first nonzero physical command"});dump("start_nonboundary_index.json",{"view_only":True,"excluded_time_indices":[0,1],"entries":non})
 resolved=json.loads((RES/"w2_p1_dataset_hashes_resolved_v2.json").read_text());actual={p:sha(REPO/p) for p in resolved["hashes"]};dump("dataset_identity_audit.json",{"manifest":str((RES/"w2_p1_dataset_hashes_resolved_v2.json").relative_to(REPO)).replace("\\","/"),"status":resolved.get("status","IMMUTABLE_RESOLVED_SOURCE_OF_TRUTH"),"hashes":actual,"all_match":actual==resolved["hashes"],"split_sha256":sha(SPLIT),"dataset_changes":0,"label_changes":0})
 pg=torch.Generator().manual_seed(20278120);pools={"BOUNDARY":fixed_pool(splits["START_RETENTION"]["train"],datasets,"boundary",8192,pg),"START_NONBOUNDARY":fixed_pool(splits["START_RETENTION"]["train"],datasets,"nonboundary",8192,pg),"STOP_RECOVERY":fixed_pool(splits["STOP_RECOVERY"]["train"],datasets,"any",8192,pg),"STEADY_STOP":fixed_pool(splits["STEADY_STOP"]["train"],datasets,"any",8192,pg)}
 for g in MOVING_GROUPS:pools[g]=fixed_pool(splits[g]["train"],datasets,"any",4096,pg)
 init=torch.load(SELECTED,map_location="cpu",weights_only=False)["actor_state_dict"];rows=[];summ=[];static_specs=[];initial_model=Student(init).to(device);grad_before=grad_analysis(initial_model,pools,device);del initial_model
 for bw in WEIGHTS:
  ow=(1-bw)/4
  for lr in LRS:
   torch.manual_seed(20278121);gen=torch.Generator().manual_seed(20278121);model=Student(init).to(device);initial=torch.cat([p.detach().cpu().flatten() for p in model.parameters()]);opt=torch.optim.Adam(model.parameters(),lr=lr);peak=0.
   for step in range(1,2001):
    def draw(key,n):
     pool=pools[key];ids=torch.randint(len(pool[0]),(n,),generator=gen);return tuple(v[ids].to(device) for v in pool)
    def loss(key,n=256):o,g,t=draw(key,n);return nn.functional.mse_loss(model(o,g),t)
    lb=loss("BOUNDARY");ls=loss("STOP_RECOVERY");lt=loss("STEADY_STOP");ln=loss("START_NONBOUNDARY");lm=torch.stack([loss(g,64) for g in MOVING_GROUPS]).mean();total=bw*lb+ow*(ls+lt+ln+lm);opt.zero_grad(set_to_none=True);total.backward();peak=max(peak,float(nn.utils.clip_grad_norm_(model.parameters(),10.0)));opt.step()
    if step in STEPS:
     ev=full_eval(model,datasets,splits,"validation",device);move=float(torch.linalg.vector_norm(torch.cat([p.detach().cpu().flatten() for p in model.parameters()])-initial));existing=all(ev[g]["pass"] for g in ("STOP_RECOVERY","STEADY_STOP","START_NONBOUNDARY",*MOVING_GROUPS));boundary=ev["BOUNDARY_COMBINED"]["pass"];spec=f"W{bw:g}_LR{lr:g}_S{step}";summary={"candidate":spec,"boundary_weight":bw,"other_group_weight":ow,"learning_rate":lr,"steps":step,"existing_static_pass":existing,"boundary_static_pass":boundary,"all_static_pass":existing and boundary,"parameter_l2_movement":move,"peak_gradient_norm":peak,"metrics":ev};summ.append(summary)
     for g,v in ev.items():rows.append({"candidate":spec,"boundary_weight":bw,"learning_rate":lr,"steps":step,"group":g,**v})
     if existing and boundary:static_specs.append(summary)
   del model,opt
 grid={"weights":list(WEIGHTS),"learning_rates":list(LRS),"checkpoints":list(STEPS),"seed":20278121,"gradient_clip":10.,"optimizer":"Adam","runs":12,"candidate_checkpoints":48,"persistent_checkpoint_writes":0};dump("resolved_probe_training_grid.json",grid);write_csv("probe_training_results.csv",rows);dump("probe_training_results.json",{"grid":grid,"candidates":summ,"static_pass_candidates":[s["candidate"] for s in static_specs]});write_csv("boundary_static_validation.csv",[r for r in rows if r["group"].startswith("BOUNDARY")]);dump("boundary_static_validation.json",{"rows":[r for r in rows if r["group"].startswith("BOUNDARY")],"static_pass_candidates":[s["candidate"] for s in static_specs]});dump("existing_group_static_validation.json",{"rows":[r for r in rows if not r["group"].startswith("BOUNDARY")]})
 # gradient after best static candidate is deterministically reconstructed in physical script; retain before values here.
 dump("start_boundary_retention_gradient_analysis.json",{"initial_step37000":grad_before,"selected_candidate_after":None,"selection_pending_physical_gate":True})
 print(json.dumps({"static_pass_candidates":len(static_specs),"names":[s["candidate"] for s in static_specs]}),flush=True)
if __name__=="__main__":main()
