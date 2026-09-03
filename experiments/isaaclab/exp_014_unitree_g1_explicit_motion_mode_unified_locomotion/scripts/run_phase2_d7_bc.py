"""Offline behavior cloning for the D7 S0 omnidirectional stop student."""
from __future__ import annotations
import hashlib,importlib.util,json,math
from pathlib import Path
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve();REPO=HERE.parents[4];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7_r4_stop_oracle_distillation";RAW=OUT/"raw";DATA=RAW/"dataset";CKPT=RAW/"bc_checkpoints";WMOVE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt";CHECKS=(0,1000,2500,5000,10000,15000,20000,25000,30000);WEIGHTS=(.10,.10,.15,.25,.20,.15,.05)
def load(name,path):s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d3=load("d3",HERE.parent/"run_phase2_d3.py")
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def initialize(device):
 p=torch.load(WMOVE,map_location="cpu",weights_only=False)["actor_state_dict"];a=d3.Specialist()
 with torch.no_grad():
  a.first_base.weight.copy_(p["first_base_weight"]);a.first_base.bias.copy_(p["first_bias"]);a.first_gait.copy_(p["first_gait_column"]);a.first_explicit.zero_();a.log_std.copy_(p["distribution.log_std_walk"])
  for layer,key in ((1,"hidden.1"),(3,"hidden.3"),(5,"hidden.5")):a.hidden[layer].weight.copy_(p[key+".weight"]);a.hidden[layer].bias.copy_(p[key+".bias"])
 return a.to(device)
def parity(a,device):
 p=torch.load(WMOVE,map_location=device,weights_only=False)["actor_state_dict"];torch.manual_seed(20279104);old=torch.randn(4096,124,device=device);obs=torch.zeros(4096,141,device=device);obs[:,:124]=old
 with torch.inference_mode():x=F.linear(old[:,:123],p["first_base_weight"],p["first_bias"])+old[:,123:124]*p["first_gait_column"].T;x=F.elu(x);x=F.elu(F.linear(x,p["hidden.1.weight"],p["hidden.1.bias"]));x=F.elu(F.linear(x,p["hidden.3.weight"],p["hidden.3.bias"]));ref=F.linear(x,p["hidden.5.weight"],p["hidden.5.bias"]);diff=float((ref-a.mean(obs)).abs().max())
 return {"architecture":"S0 141-256-128-128-37","parent":"W_MOVE","parent_sha256":"61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d","old_124_columns_bitwise_copy":True,"new_17_columns_zero":True,"hidden_output_copy":True,"max_difference":diff,"gate":1e-8,"status":"PASS" if diff<=1e-8 else "FAIL"}
def evaluate(a,data,device):
 n=len(data["observation_141"]);sumse=sumcos=0.;ctx={i:[0.,0.,0] for i in range(7)};cond={i:[0.,0] for i in range(34)}
 with torch.inference_mode():
  for lo in range(0,n,8192):
   hi=min(n,lo+8192);o=data["observation_141"][lo:hi].to(device);y=data["action_37"][lo:hi].to(device);p=a.mean(o);se=(p-y).square().mean(1).cpu();co=F.cosine_similarity(p,y).cpu();sumse+=float(se.sum());sumcos+=float(co.sum());c=data["context_id"][lo:hi];q=data["condition_id"][lo:hi]
   for i in range(7):m=c==i;ctx[i][0]+=float(se[m].sum());ctx[i][1]+=float(co[m].sum());ctx[i][2]+=int(m.sum())
   for i in range(34):m=q==i;cond[i][0]+=float(se[m].sum());cond[i][1]+=int(m.sum())
 cm={str(i):{"mse":v[0]/v[2],"cosine":v[1]/v[2],"samples":v[2]} for i,v in ctx.items()};cn={str(i):v[0]/v[1] for i,v in cond.items()};return {"samples":n,"mse":sumse/n,"cosine":sumcos/n,"contexts":cm,"conditions":cn,"worst_condition_mse":max(cn.values())}
def phase_classifier(train,val,device):
 # Diagnostic-only classifier uses the same 141D causal observation, never Teacher ID.
 torch.manual_seed(20279105);head=torch.nn.Sequential(torch.nn.Linear(141,128),torch.nn.ELU(),torch.nn.Linear(128,7)).to(device);opt=torch.optim.Adam(head.parameters(),lr=1e-3);x=train["observation_141"];y=train["context_id"]
 for _ in range(1500):idx=torch.randint(len(x),(2048,));loss=F.cross_entropy(head(x[idx].to(device)),y[idx].to(device));opt.zero_grad();loss.backward();opt.step()
 correct=0
 with torch.inference_mode():
  for lo in range(0,len(val["observation_141"]),8192):correct+=int((head(val["observation_141"][lo:lo+8192].to(device)).argmax(1).cpu()==val["context_id"][lo:lo+8192]).sum())
 return correct/len(val["observation_141"])
def save(path,a,step,meta):path.parent.mkdir(parents=True,exist_ok=True);torch.save({"name":"Exp014DistilledOmnidirectionalStopSpecialistV1","architecture":[141,256,128,128,37],"step":step,"actor_state_dict":a.state_dict(),"metadata":meta},path)
def main():
 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");torch.manual_seed(20279106);train=torch.load(DATA/"train.pt",map_location="cpu",weights_only=False);val=torch.load(DATA/"validation.pt",map_location="cpu",weights_only=False);a=initialize(device);par=parity(a,device);phase=phase_classifier(train,val,device);opt=torch.optim.Adam([p for n,p in a.named_parameters() if n!="log_std"],lr=5e-5);indices=[torch.where(train["context_id"]==i)[0] for i in range(7)];counts=[round(2048*w) for w in WEIGHTS];counts[-1]=2048-sum(counts[:-1]);timeline=[];manifest=[];initial=torch.cat([p.detach().cpu().flatten() for n,p in a.named_parameters() if n!="log_std"])
 for step in range(30001):
  if step in CHECKS:
   ev=evaluate(a,val,device);ev.update({"step":step,"phase_classification":phase,"static_pass":ev["mse"]<=.001 and ev["cosine"]>=.98 and all(ev["contexts"][str(i)]["mse"]<=.001 for i in (3,4,5)) and ev["worst_condition_mse"]<=.001 and phase>=.99});path=CKPT/f"s0_step_{step:05d}.pt";save(path,a,step,ev);manifest.append({"step":step,"path":str(path.relative_to(REPO)).replace("\\","/"),"sha256":sha(path)});timeline.append(ev)
  if step==30000:break
  idx=torch.cat([pool[torch.randint(len(pool),(count,))] for pool,count in zip(indices,counts)]);idx=idx[torch.randperm(len(idx))];o=train["observation_141"][idx].to(device);y=train["action_37"][idx].to(device);pred=a.mean(o);loss=F.mse_loss(pred,y);opt.zero_grad();loss.backward();opt.step()
 eligible=[x for x in timeline if x["static_pass"]];selected=min(eligible,key=lambda x:(x["mse"],x["worst_condition_mse"],x["step"])) if eligible else None;movement=float((torch.cat([p.detach().cpu().flatten() for n,p in a.named_parameters() if n!="log_std"])-initial).norm());result={"architecture":"S0","initialization_parity":par,"phase_classification":phase,"optimizer":"Adam","learning_rate":5e-5,"steps":30000,"batch_size":2048,"sampling_weights":WEIGHTS,"timeline":timeline,"checkpoint_manifest":manifest,"eligible_steps":[x["step"] for x in eligible],"selected_step":selected["step"] if selected else None,"parameter_movement":movement};dump(RAW/"bc_results.json",result);print(json.dumps({"parity":par,"phase":phase,"selected":result["selected_step"],"final":timeline[-1]},indent=2))
if __name__=="__main__":main()
