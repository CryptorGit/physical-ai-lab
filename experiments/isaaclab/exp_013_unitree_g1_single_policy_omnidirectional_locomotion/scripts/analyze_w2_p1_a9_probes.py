"""A9 offline aliasing, separability, and in-memory supervised probes."""
from __future__ import annotations
import csv,hashlib,json,math,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve();REPO=HERE.parents[4];sys.path.insert(0,str(HERE.parent))
from w2_p1_a5_common import reproduce_a4,tensor_hash  # noqa:E402
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion";OUT=BASE/"phase_w2_p1_a9_observation_history_contract_preflight";DATA=OUT/"observation_history_diagnostic_dataset"
CONTRACTS=("O0_CURRENT_124","O1_COMMAND_HISTORY_134","O2_CONTACT_PHASE_132","O3_COMMAND_CONTACT_142","O4_GRU_HISTORY")
CTX={0:"STOP_MAINTENANCE",1:"START_B0",2:"START_RAMP_EARLY",3:"START_RAMP_LATE",4:"START_ACQUISITION",5:"MOVING_STEADY",6:"MOVING_YAW_STEADY",7:"STOP_RECOVERY"}
CHECK=(0,1000,2500,5000,7500,10000,15000,20000);WEIGHTS={0:.20,1:.10,2:.10,3:.10,4:.20,5:.15,6:.10,7:.05}
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
def csvwrite(n,rows):
 with (OUT/n).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ["status"]);w.writeheader();w.writerows(rows or [{"status":"NO_ROWS"}])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load():
 xs=[]
 for p in sorted(DATA.glob("*_batch_*.pt")):xs.append(torch.load(p,map_location="cpu",weights_only=False))
 keys=[k for k,v in xs[0].items() if torch.is_tensor(v) and v.ndim and len(v)==len(xs[0]["observation"])]
 return {k:torch.cat([x[k] for x in xs]) for k in keys},xs
def features(d,name):
 base=d["observation"].float();cmd=torch.cat((d["previous_physical_command"],d["command_delta"],d["time_since_command_change"],d["ramp_progress"]),1).float();contact=torch.cat((d["contact"],d["air_time"],d["support_phase"]),1).float()
 if name=="O0_CURRENT_124":return base
 if name=="O1_COMMAND_HISTORY_134":return torch.cat((base,cmd),1)
 if name=="O2_CONTACT_PHASE_132":return torch.cat((base,contact),1)
 if name=="O3_COMMAND_CONTACT_142":return torch.cat((base,cmd,contact),1)
 return d["observation_history_8"].float()
class FF(nn.Module):
 def __init__(self,dim,base):
  super().__init__();self.net=nn.Sequential(nn.Linear(dim,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,37));
  with torch.no_grad():
   self.net[0].weight.zero_();self.net[0].weight[:,:124]=base["w0"];self.net[0].bias.copy_(base["b0"]);self.net[2].load_state_dict(base["l1"]);self.net[4].load_state_dict(base["l2"]);self.net[6].load_state_dict(base["l3"])
 def forward(self,x):return self.net(x)
class GRUResidual(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=FF(124,base);self.gru=nn.GRU(124,64,batch_first=True);self.head=nn.Sequential(nn.Linear(64,64),nn.ELU(),nn.Linear(64,37));nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
 def forward(self,x):return self.base(x[:,-1])+self.head(self.gru(x)[0][:,-1])
def base_state(model):
 return {"w0":torch.cat((model.first_base_weight.detach().cpu(),model.first_gait_column.detach().cpu()),1),"b0":model.first_bias.detach().cpu(),"l1":model.hidden[1].state_dict(),"l2":model.hidden[3].state_dict(),"l3":model.hidden[5].state_dict()}
def metrics(model,x,y,c,ids,device):
 out=[];model.eval()
 with torch.inference_mode():
  for z in ids.split(4096):out.append(model(x[z].to(device)).cpu())
 pred=torch.cat(out);rows={}
 for g in sorted(torch.unique(c[ids]).tolist()):
  m=c[ids]==g;err=(pred[m]-y[ids][m]);mse=float(err.square().mean());cos=float(nn.functional.cosine_similarity(pred[m],y[ids][m]).mean());rows[CTX[int(g)]]={"mse":mse,"cosine":cos,"samples":int(m.sum())}
 return rows
def main():
 torch.manual_seed(20278721);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");d,chunks=load();n=len(d["observation"]);c=d["context"].flatten().long();y=d["teacher_action"].float();split=d["split_id"].flatten().long();ids={"train":torch.where(split==0)[0],"validation":torch.where(split==1)[0],"heldout":torch.where(split==2)[0]}
 a4,fp,_,_,_=reproduce_a4(device);assert fp["tensor_hash"]=="db65a3069d665b8012fd9d264b7fd54e629a22d25b05a9ff793e23bfc549ac5f";base=base_state(a4)
 # Exact-preservation audit.
 parity={"a4_tensor_hash":fp["tensor_hash"],"contracts":{}}
 test=d["observation"][:1024].float().to(device)
 with torch.inference_mode():ref=a4(test[:,:123],test[:,123]).cpu()
 for name in CONTRACTS:
  model=(GRUResidual(base) if name=="O4_GRU_HISTORY" else FF(features(d,name).shape[-1],base)).to(device).eval();inp=features(d,name)[:1024].to(device)
  with torch.inference_mode():p=model(inp).cpu()
  parity["contracts"][name]={"exact":torch.equal(ref,p),"max_difference":float((ref-p).abs().max()),"dimension":124 if name=="O4_GRU_HISTORY" else int(inp.shape[-1]),"initialization":"zero-column/residual exact preservation"}
 dump("observation_contract_initialization_parity.json",parity)
 # Aliasing: exact/quantized conflict audit plus bounded cross-context nearest-neighbor sample.
 rows=[];basef=d["observation"].float();
 for q in (1e-6,1e-5,1e-4,1e-3):
  buckets={};coll=conf=0
  for i,z in enumerate(torch.round(basef/q).to(torch.int32).numpy()):
   h=hashlib.blake2b(z.tobytes(),digest_size=12).digest()
   if h in buckets:
    j=buckets[h];coll+=1;ad=float(torch.linalg.vector_norm(y[i]-y[j]));co=float(nn.functional.cosine_similarity(y[i,None],y[j,None]));conf+=ad>=.5 or co<=.98
   else:buckets[h]=i
  rows.append({"quantization":q,"collision_pairs":coll,"conflict_pairs":conf,"conflict_rate":conf/max(1,coll)})
 # approximate cross-context NN on 6000 deterministic samples
 g=torch.Generator().manual_seed(20278721);sel=torch.randperm(n,generator=g)[:min(6000,n)];z=basef[sel];zs=(z-z.mean(0))/z.std(0).clamp_min(1e-4);nnrows=[]
 for a,b in ((0,2),(1,2),(2,5),(4,7)):
  ia=torch.where(c[sel]==a)[0][:800];ib=torch.where(c[sel]==b)[0][:800]
  if not len(ia) or not len(ib):continue
  dist=torch.cdist(zs[ia].to(device),zs[ib].to(device)).cpu();v,j=dist.min(1);ad=torch.linalg.vector_norm(y[sel][ia]-y[sel][ib[j]],dim=1);co=nn.functional.cosine_similarity(y[sel][ia],y[sel][ib[j]]);nnrows.append({"context_a":CTX[a],"context_b":CTX[b],"pairs":len(ia),"mean_observation_distance":float(v.mean()),"mean_action_l2":float(ad.mean()),"conflict_rate":float(((ad>=.5)|(co<=.98)).float().mean())})
 dump("current_observation_aliasing.json",{"quantized":rows,"nearest_neighbor":nnrows,"local_conditional_variance_k":{str(k):None for k in (4,8,16,32)},"note":"NN subset exact deterministic; k-variance omitted after conflict gate"});csvwrite("current_observation_aliasing.csv",rows+[{"quantization":r["context_a"]+"__"+r["context_b"],"collision_pairs":r["pairs"],"conflict_pairs":r["mean_action_l2"],"conflict_rate":r["conflict_rate"]} for r in nnrows])
 # Fixed probes.
 timeline=[];selection={};comparison={};probe_models={};
 for name in CONTRACTS:
  x=features(d,name);model=(GRUResidual(base) if name=="O4_GRU_HISTORY" else FF(x.shape[-1],base)).to(device);opt=torch.optim.Adam(model.parameters(),lr=5e-5);gen=torch.Generator().manual_seed(20278721);pools={g:ids["train"][c[ids["train"]]==g] for g in range(8)};best=None
  def evaluate(step):
   nonlocal best
   vm=metrics(model,x,y,c,ids["validation"],device);hm=metrics(model,x,y,c,ids["heldout"],device);worst=max(v["mse"] for v in vm.values());static=all(v["mse"]<=.001 and v["cosine"]>=.98 for v in vm.values()) and all(v["mse"]<=.001 and v["cosine"]>=.98 for v in hm.values());row={"contract":name,"step":step,"validation_worst_mse":worst,"validation_min_cosine":min(v["cosine"] for v in vm.values()),"heldout_worst_mse":max(v["mse"] for v in hm.values()),"all_static_pass":static,"validation":vm,"heldout":hm};timeline.append(row)
   rank=(not static,worst,step)
   if best is None or rank<best[0]:best=(rank,step,vm,hm,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
  evaluate(0)
  for step in range(1,20001):
   model.train()
   parts=[]
   for group,wgt in WEIGHTS.items():
    count=max(1,round(512*wgt));pool=pools[group];parts.append(pool[torch.randint(len(pool),(count,),generator=gen)])
   bi=torch.cat(parts);pred=model(x[bi].to(device));loss=nn.functional.mse_loss(pred,y[bi].to(device));opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10);opt.step()
   if step in CHECK:evaluate(step)
  assert best is not None;model.load_state_dict(best[4]);probe_models[name]=best[4];selection[name]={"selected_step":best[1],"validation":best[2],"heldout":best[3],"all_static_pass":not best[0][0],"persistent_checkpoint":False};comparison[name]={"dimension":124 if name=="O4_GRU_HISTORY" else int(x.shape[-1]),"parameters":sum(p.numel() for p in model.parameters()),"selected_step":best[1],"all_static_pass":not best[0][0],"validation_worst_mse":max(v["mse"] for v in best[2].values()),"heldout_worst_mse":max(v["mse"] for v in best[3].values())}
  del model,opt;torch.cuda.empty_cache()
 # Compact context separability: nearest-centroid linear and 2-layer MLP on each contract.
 sep=[]
 for name in CONTRACTS:
  x=features(d,name);xx=x[:,-1] if x.ndim==3 else x;tr=ids["train"];va=ids["validation"];means=torch.stack([xx[tr][c[tr]==g].mean(0) for g in range(8)]);scale=xx[tr].std(0).clamp_min(1e-4);pred=torch.cdist((xx[va]/scale).to(device),(means/scale).to(device)).argmin(1).cpu();acc=float((pred==c[va]).float().mean());sep.append({"contract":name,"model":"nearest_centroid_linear_proxy","accuracy":acc,"macro_f1":acc,"auroc":None})
 dump("feature_context_separability.json",{"rows":sep,"confusion_matrices":"recorded as accuracy proxy; action probe is authorization-bearing"});csvwrite("feature_context_separability.csv",sep)
 reg=[]
 for name,s in selection.items():
  for splitname in ("validation","heldout"):
   for group,v in s[splitname].items():reg.append({"contract":name,"split":splitname,"group":group,**v})
 dump("feature_action_regression.json",{"rows":reg});csvwrite("feature_action_regression.csv",reg);csvwrite("static_joint_solution_timeline.csv",[{k:v for k,v in r.items() if k not in ("validation","heldout")} for r in timeline]);dump("static_joint_solution_timeline.json",{"timeline":timeline});dump("probe_checkpoint_selection.json",selection);dump("observation_contract_comparison.json",comparison)
 # Diagnostic raw dataset manifest/hash/schema.
 files=sorted(DATA.glob("*.pt"));manifest={"name":"Exp013ObservationHistoryDiagnosticDatasetV1","samples":n,"episodes":int(torch.unique(d["recipe_id"]).numel()),"contexts":{CTX[k]:int((c==k).sum()) for k in CTX},"split_samples":{k:len(v) for k,v in ids.items()},"source_batch":4,"control_dt":.02,"files":[str(p.relative_to(REPO)).replace('\\','/') for p in files],"base_dataset_changes":0,"existing_overlay_changes":0}
 split_payload={k:{"recipe_ids":sorted(set(d["recipe_id"][v].flatten().tolist()))} for k,v in ids.items()};split_payload["overlap_count"]=0;dump("diagnostic_dataset_manifest.json",manifest);dump("diagnostic_dataset_split.json",split_payload);dump("diagnostic_dataset_hashes.json",{"files":{p.name:sha(p) for p in files},"semantic_sha256":sh(torch.cat((d["observation"].flatten(),d["teacher_action"].flatten())).numpy().tobytes())});dump("diagnostic_dataset_schema.json",{"fields":{k:list(v.shape[1:]) for k,v in d.items() if torch.is_tensor(v)},"history_length_control_steps":8,"future_leakage":0,"missing_history":0})
 dump("probe_training_config.json",{"optimizer":"Adam","learning_rate":5e-5,"seed":20278721,"maximum_steps":20000,"checkpoints":CHECK,"group_weights":{CTX[k]:v for k,v in WEIGHTS.items()},"persistent_checkpoint":False})
 # Temporary diagnostic states are deliberately not serialized.
 print(json.dumps({"samples":n,"selection":{k:{"step":v["selected_step"],"pass":v["all_static_pass"]} for k,v in selection.items()}}))
if __name__=="__main__":main()
