"""Retrain S0 on immutable base data plus cumulative DAgger additions."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
HERE=Path(__file__).resolve();EXP=HERE.parent.parent;REPO=EXP.parents[2];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion";BASE=OUT/"phase1_dataset";DAG=OUT/"dagger_dataset";CP=OUT/"dagger_checkpoints"
sys.path.insert(0,str(EXP/"src"));sys.path.insert(0,str(HERE.parent));from g1_explicit_motion_mode.student import ExplicitModeStudent,checkpoint_payload;from train_static import metrics,GROUPS,WEIGHTS
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--round",type=int,required=True);p.add_argument("--parent",required=True);p.add_argument("--steps",type=int,default=10000);p.add_argument("--learning-rate",type=float,default=5e-5);a=p.parse_args();chunks=[torch.load(x,map_location="cpu",weights_only=False) for x in sorted(BASE.glob("*.pt"))]+[torch.load(DAG/f"round_{r}.pt",map_location="cpu",weights_only=False) for r in range(1,a.round+1)];keys=("observation_141","teacher_action","context","group","condition_index","recipe_id","control_step","split_id");d={k:torch.cat([x[k] for x in chunks]) for k in keys};device=torch.device("cuda" if torch.cuda.is_available() else "cpu");payload=torch.load(a.parent,map_location=device,weights_only=False);model=ExplicitModeStudent(tuple(payload["architecture"][1:-1])).to(device);model.load_state_dict(payload["actor_state_dict"]);opt=torch.optim.Adam(model.parameters(),lr=a.learning_rate);train=torch.nonzero(d["split_id"].flatten()==0).flatten();val=torch.nonzero(d["split_id"].flatten()==1).flatten();groups=[train[d["group"][train].flatten()==i] for i in range(6)];counts=[round(1024*w) for w in WEIGHTS];counts[-1]+=1024-sum(counts);gen=torch.Generator().manual_seed(20261100+a.round);CP.mkdir(parents=True,exist_ok=True);timeline=[]
 for step in range(1,a.steps+1):
  sel=[g[torch.randint(len(g),(c,),generator=gen)] for g,c in zip(groups,counts)];ix=torch.cat(sel)[torch.randperm(1024,generator=gen)];x=d["observation_141"][ix].to(device);y=d["teacher_action"][ix].to(device);loss=torch.nn.functional.mse_loss(model(x),y);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
  if step in (1000,2500,5000,10000):
   m=metrics(model.eval(),d,val,device);path=CP/f"round_{a.round}_step_{step:05d}.pt";torch.save(checkpoint_payload(model,dagger_round=a.round,step=step,optimizer_state_dict=opt.state_dict(),validation=m,parent=a.parent),path);timeline.append({"step":step,"checkpoint":path.relative_to(REPO).as_posix(),"sha256":sha(path),"validation":m});print(json.dumps({"step":step,"mse":m["aggregate_mse"],"worst":m["worst_condition_mse"],"pass":m["static_gate_pass"]}),flush=True);model.train()
 passing=[r for r in timeline if r["validation"]["static_gate_pass"]];selected=min(passing,key=lambda r:(r["validation"]["worst_condition_mse"],r["step"])) if passing else min(timeline,key=lambda r:(r["validation"]["worst_condition_mse"],r["step"]));result={"round":a.round,"parent":a.parent,"cumulative_samples":len(d["observation_141"]),"added_samples":len(chunks[-1]["observation_141"]),"timeline":timeline,"selected":selected,"heldout_not_reopened":True};(OUT/f"dagger_round_{a.round}_training.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
