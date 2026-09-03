"""Validation-selected, group-balanced EXP 014 static capacity probe."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
import torch

HERE=Path(__file__).resolve();EXP=HERE.parent.parent;REPO=EXP.parents[2];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion";RAW=OUT/"phase1_dataset";CP=OUT/"static_checkpoints"
WALK=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
sys.path.insert(0,str(EXP/"src"))
from g1_explicit_motion_mode.student import ARCHITECTURES,checkpoint_payload,initialize_s0_from_w1b,widen_student,ExplicitModeStudent

CHECKPOINTS=(0,1000,2500,5000,10000,15000,20000,25000,30000);WEIGHTS=(.15,.25,.20,.15,.20,.05);GROUPS=("STAND_HOLD","STAND_TO_WALK","WALK_STEADY","WALK_YAW","WALK_TO_STAND","STAND_AFTER_STOP");CONTEXTS=("STAND_HOLD","STAND_TO_WALK_B0","STAND_TO_WALK_RAMP","STAND_TO_WALK_ACQUISITION","WALK_STEADY","WALK_PURE_YAW","WALK_MOVING_YAW","WALK_TO_STAND_DECELERATION","WALK_TO_STAND_RECOVERY","STAND_AFTER_STOP")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def load():
 chunks=[torch.load(p,map_location="cpu",weights_only=False) for p in sorted(RAW.glob("phase1_batch_*.pt"))];keys=("observation_141","teacher_action","context","group","condition_index","recipe_id","control_step","split_id")
 return {k:torch.cat([d[k] for d in chunks]) for k in keys}
def batches(indices,size=8192):
 for i in range(0,len(indices),size):yield indices[i:i+size]
@torch.inference_mode()
def metrics(model,d,indices,device):
 pred=[]
 for b in batches(indices):pred.append(model(d["observation_141"][b].to(device)).cpu())
 pred=torch.cat(pred);target=d["teacher_action"][indices];ctx=d["context"][indices].flatten();grp=d["group"][indices].flatten();cond=d["condition_index"][indices].flatten();err=(pred-target).square().mean(1);cos=torch.nn.functional.cosine_similarity(pred,target,dim=1)
 def summary(mask):return {"samples":int(mask.sum()),"mse":float(err[mask].mean()),"cosine":float(cos[mask].mean())} if mask.any() else {"samples":0,"mse":None,"cosine":None}
 groups={name:summary(grp==i) for i,name in enumerate(GROUPS)};contexts={name:summary(ctx==i) for i,name in enumerate(CONTEXTS)};directions={str(i):summary(cond==i) for i in torch.unique(cond).tolist()}
 worst_condition=max(v["mse"] for v in directions.values());mandatory=[*groups.values(),*[contexts[CONTEXTS[i]] for i in (0,1,2,3,4,6,7,8)]]
 gate=all(v["mse"]<=.001 and v["cosine"]>=.98 for v in mandatory) and worst_condition<=.001
 # Mode-conditioned dual-state classification at the exact B0 recipe pairs.
 index_pos={int(global_i):j for j,global_i in enumerate(indices.tolist())};correct=total=0
 recipes=torch.unique(d["recipe_id"][indices])
 for r in recipes.tolist():
  stand=torch.nonzero((d["recipe_id"].flatten()==r)&(d["context"].flatten()==0)&(d["control_step"].flatten()==0)).flatten();walk=torch.nonzero((d["recipe_id"].flatten()==r)&(d["context"].flatten()==1)).flatten()
  if not len(stand) or not len(walk) or int(stand[0]) not in index_pos or int(walk[0]) not in index_pos:continue
  s=index_pos[int(stand[0])];w=index_pos[int(walk[0])];sl=target[s];wl=target[w]
  correct+=int(torch.linalg.vector_norm(pred[s]-sl)<torch.linalg.vector_norm(pred[s]-wl));correct+=int(torch.linalg.vector_norm(pred[w]-wl)<torch.linalg.vector_norm(pred[w]-sl));total+=2
 accuracy=correct/total if total else 0.;gate=gate and accuracy>=.99
 return {"aggregate_mse":float(err.mean()),"aggregate_cosine":float(cos.mean()),"groups":groups,"contexts":contexts,"conditions":directions,"worst_condition_mse":worst_condition,"dual_mode_classification":accuracy,"static_gate_pass":gate}
def init(name,device,d,train_idx):
 s0=initialize_s0_from_w1b(WALK).to(device)
 if name=="S0":return s0,{"method":"bitwise W1B copy","parity_steps":0}
 model=widen_student(s0,name).to(device);model.train();opt=torch.optim.Adam(model.parameters(),lr=5e-5);gen=torch.Generator().manual_seed(20260804)
 # Registered fallback: short same-dataset W1B/S0 parity distillation.
 for _ in range(500):
  ix=train_idx[torch.randint(len(train_idx),(1024,),generator=gen)];x=d["observation_141"][ix].to(device)
  with torch.no_grad():y=s0(x)
  loss=torch.nn.functional.mse_loss(model(x),y);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
 model.eval();sample=d["observation_141"][train_idx[:4096]].to(device)
 with torch.inference_mode():diff=float((model(sample)-s0(sample)).abs().max())
 return model,{"method":"FUNCTION_PRESERVING_WIDENING_UNAVAILABLE; 500-step W1B parity distillation","parity_steps":500,"post_distillation_max_abs_difference":diff}
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",choices=tuple(ARCHITECTURES),required=True);p.add_argument("--batch-size",type=int,default=1024);p.add_argument("--learning-rate",type=float,default=5e-5);p.add_argument("--gradient-clip",type=float,default=10.);p.add_argument("--max-steps",type=int,default=30000);a=p.parse_args();torch.manual_seed(20260803);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");d=load();train_idx=torch.nonzero(d["split_id"].flatten()==0).flatten();val_idx=torch.nonzero(d["split_id"].flatten()==1).flatten();held_idx=torch.nonzero(d["split_id"].flatten()==2).flatten();group_idx=[train_idx[d["group"][train_idx].flatten()==i] for i in range(6)];model,initialization=init(a.model,device,d,train_idx);opt=torch.optim.Adam(model.parameters(),lr=a.learning_rate);gen=torch.Generator().manual_seed(20260803);CP.mkdir(parents=True,exist_ok=True);timeline=[];start=time.time()
 def evaluate(step):
  model.eval();m=metrics(model,d,val_idx,device);path=CP/f"{a.model.lower()}_step_{step:05d}.pt";torch.save(checkpoint_payload(model,model_name=a.model,step=step,optimizer_state_dict=opt.state_dict(),validation=m,initialization=initialization,dataset_hashes=json.loads((OUT/"dataset_hashes.json").read_text())),path);row={"model":a.model,"step":step,"elapsed_s":time.time()-start,"checkpoint":path.relative_to(REPO).as_posix(),"sha256":sha(path),"parameter_count":model.parameter_count,"validation":m};timeline.append(row);print(json.dumps({"step":step,"aggregate_mse":m["aggregate_mse"],"worst_condition_mse":m["worst_condition_mse"],"dual":m["dual_mode_classification"],"pass":m["static_gate_pass"]}),flush=True);model.train()
 evaluate(0)
 counts=[round(a.batch_size*w) for w in WEIGHTS];counts[-1]+=a.batch_size-sum(counts)
 for step in range(1,a.max_steps+1):
  selections=[idx[torch.randint(len(idx),(count,),generator=gen)] for idx,count in zip(group_idx,counts)];ix=torch.cat(selections)[torch.randperm(a.batch_size,generator=gen)];x=d["observation_141"][ix].to(device);y=d["teacher_action"][ix].to(device);loss=torch.nn.functional.mse_loss(model(x),y);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),a.gradient_clip);opt.step()
  if step in CHECKPOINTS[1:]:evaluate(step)
 # Validation-only selection: pass, worst metric, then earlier checkpoint.
 passing=[r for r in timeline if r["validation"]["static_gate_pass"]];selected=min(passing,key=lambda r:(r["validation"]["worst_condition_mse"],r["step"])) if passing else min(timeline,key=lambda r:(r["validation"]["worst_condition_mse"],r["step"]))
 # Held-out is opened exactly once after the selection is immutable.
 payload=torch.load(REPO/selected["checkpoint"],map_location=device,weights_only=False);fixed=ExplicitModeStudent(ARCHITECTURES[a.model]).to(device);fixed.load_state_dict(payload["actor_state_dict"]);held=metrics(fixed.eval(),d,held_idx,device);result={"model":a.model,"initialization":initialization,"timeline":timeline,"selection_rule":"validation only; mandatory PASS then best worst-condition MSE then earlier checkpoint","selected":selected,"heldout_once_after_freeze":held,"heldout_used_for_selection":False,"capacity_classification":"STATIC_PASS" if selected["validation"]["static_gate_pass"] else f"{a.model}_CAPACITY_FAIL"}
 (OUT/f"capacity_{a.model.lower()}.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"selected":selected,"heldout":held,"classification":result["capacity_classification"]},indent=2))
if __name__=="__main__":main()
