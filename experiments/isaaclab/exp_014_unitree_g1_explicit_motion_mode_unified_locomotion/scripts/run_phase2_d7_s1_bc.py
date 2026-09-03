"""Function-preserving S0-to-S1 widening and BC after S0 static capacity failure."""
from __future__ import annotations
import hashlib,importlib.util,json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
HERE=Path(__file__).resolve();REPO=HERE.parents[4];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7_r4_stop_oracle_distillation";RAW=OUT/"raw";DATA=RAW/"dataset";CKPT=RAW/"bc_checkpoints";S0=CKPT/"s0_step_30000.pt";CHECKS=(0,1000,2500,5000,10000,15000,20000,25000,30000);WEIGHTS=(.10,.10,.15,.25,.20,.15,.05)
def load(name,path):s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
bc=load("bc",HERE.parent/"run_phase2_d7_bc.py");d3=bc.d3
class S1(nn.Module):
 def __init__(self):super().__init__();self.mlp=nn.Sequential(nn.Linear(141,512),nn.ELU(),nn.Linear(512,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,37))
 def mean(self,x):return self.mlp(x)
def widen(s0,device):
 s=S1().to(device);W1=torch.cat((s0.first_base.weight,s0.first_gait,s0.first_explicit),1);b1=s0.first_base.bias;W2=s0.hidden[1].weight;b2=s0.hidden[1].bias;W3=s0.hidden[3].weight;b3=s0.hidden[3].bias;W4=s0.hidden[5].weight;b4=s0.hidden[5].bias
 with torch.no_grad():
  # Net2Wider: duplicate layer-1 twice, layer-2 four times, layer-3 twice.
  s.mlp[0].weight.copy_(W1.repeat_interleave(2,0));s.mlp[0].bias.copy_(b1.repeat_interleave(2));w2=W2.repeat_interleave(4,0).repeat_interleave(2,1)/2;s.mlp[2].weight.copy_(w2);s.mlp[2].bias.copy_(b2.repeat_interleave(4));w3=W3.repeat_interleave(2,0).repeat_interleave(4,1)/4;s.mlp[4].weight.copy_(w3);s.mlp[4].bias.copy_(b3.repeat_interleave(2));s.mlp[6].weight.copy_(W4.repeat_interleave(2,1)/2);s.mlp[6].bias.copy_(b4)
 return s
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def save(path,a,step,meta):path.parent.mkdir(parents=True,exist_ok=True);torch.save({"name":"Exp014DistilledOmnidirectionalStopSpecialistV1","architecture":[141,512,512,256,37],"step":step,"actor_state_dict":a.state_dict(),"metadata":meta},path)
def phase_classifier(train,val,device):
 torch.manual_seed(20279107);head=nn.Sequential(nn.Linear(141,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,7)).to(device);opt=torch.optim.Adam(head.parameters(),lr=1e-3);pools=[torch.where(train["context_id"]==i)[0] for i in range(7)]
 for _ in range(3000):idx=torch.cat([p[torch.randint(len(p),(256,))] for p in pools]);loss=F.cross_entropy(head(train["observation_141"][idx].to(device)),train["context_id"][idx].to(device));opt.zero_grad();loss.backward();opt.step()
 correct=0
 with torch.inference_mode():
  for lo in range(0,len(val["observation_141"]),8192):correct+=int((head(val["observation_141"][lo:lo+8192].to(device)).argmax(1).cpu()==val["context_id"][lo:lo+8192]).sum())
 return correct/len(val["observation_141"])
def main():
 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");torch.manual_seed(20279108);train=torch.load(DATA/"train.pt",map_location="cpu",weights_only=False);val=torch.load(DATA/"validation.pt",map_location="cpu",weights_only=False);p=torch.load(S0,map_location=device,weights_only=False);s0=d3.Specialist().to(device);s0.load_state_dict(p["actor_state_dict"]);a=widen(s0,device);test=val["observation_141"][:4096].to(device)
 with torch.inference_mode():diff=float((a.mean(test)-s0.mean(test)).abs().max())
 widening={"source":"S0 step 30000","source_sha256":sha(S0),"architecture":[141,512,512,256,37],"method":"Net2Wider duplicate/split outgoing weights","samples":len(test),"max_difference":diff,"function_preserving":diff<=1e-5};phase=phase_classifier(train,val,device);opt=torch.optim.Adam(a.parameters(),lr=5e-5);indices=[torch.where(train["context_id"]==i)[0] for i in range(7)];counts=[round(2048*w) for w in WEIGHTS];counts[-1]=2048-sum(counts[:-1]);timeline=[];manifest=[];initial=torch.cat([x.detach().cpu().flatten() for x in a.parameters()])
 for step in range(30001):
  if step in CHECKS:
   ev=bc.evaluate(a,val,device);ev.update({"step":step,"phase_classification":phase,"static_pass":ev["mse"]<=.001 and ev["cosine"]>=.98 and all(ev["contexts"][str(i)]["mse"]<=.001 for i in (3,4,5)) and ev["worst_condition_mse"]<=.001 and phase>=.99});path=CKPT/f"s1_step_{step:05d}.pt";save(path,a,step,ev);manifest.append({"step":step,"path":str(path.relative_to(REPO)).replace("\\","/"),"sha256":sha(path)});timeline.append(ev)
  if step==30000:break
  idx=torch.cat([pool[torch.randint(len(pool),(count,))] for pool,count in zip(indices,counts)]);idx=idx[torch.randperm(len(idx))];o=train["observation_141"][idx].to(device);y=train["action_37"][idx].to(device);loss=F.mse_loss(a.mean(o),y);opt.zero_grad();loss.backward();opt.step()
 eligible=[x for x in timeline if x["static_pass"]];selected=min(eligible,key=lambda x:(x["mse"],x["worst_condition_mse"],x["step"])) if eligible else None;result={"architecture":"S1","widening":widening,"phase_classification":phase,"timeline":timeline,"checkpoint_manifest":manifest,"eligible_steps":[x["step"] for x in eligible],"selected_step":selected["step"] if selected else None,"parameter_movement":float((torch.cat([x.detach().cpu().flatten() for x in a.parameters()])-initial).norm())};bc.dump(RAW/"s1_bc_results.json",result);print(json.dumps({"widening":widening,"phase":phase,"selected":result["selected_step"],"final":timeline[-1]},indent=2))
if __name__=="__main__":main()
