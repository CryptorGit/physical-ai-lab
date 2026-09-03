"""D23 train-only explicit first-swing-foot staged reachability audit.

This runner creates no policy checkpoint and never reads validation/held-out data.
The only learned object is an optional in-memory diagnostic distillation clone.
"""
from __future__ import annotations

import argparse, copy, hashlib, importlib.util, json, math, random, sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d23_explicit_lead_start_reachability"; RAW=OUT/"raw"
D16=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d16_dedicated_start_specialist"
D6=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d6_omnidirectional_stop_teacher"
MIRROR=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_d1_yaw_translation_interference_diagnosis/robot_mirror_contract.json"
DT=.02; SEED=20279701

def mod(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
d22=mod("d22_d23",HERE.parent/"run_phase2_d22_direct_preflight.py");d17=d22.d17;d16=d22.d16;d15=d22.d15;d3=d22.d3;d6=d22.d6
from g1_explicit_motion_mode.contract import minimum_jerk
from g1_omnidirectional.policy import FrozenGaitActor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli

def f(x):return float(x.detach().cpu()) if torch.is_tensor(x) else float(x)
def sha_tensor(x):return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
def dump(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")

class Direct143(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(143,512),nn.ELU(),nn.Linear(512,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,37));self.log_std=nn.Parameter(torch.full((37,),-1.));self.parity=False
 def raw(self,x):
  if not self.parity:return self.net(x)
  z=torch.nn.functional.linear(x[:,:141],self.net[0].weight[:,:141].contiguous(),self.net[0].bias);z=torch.nn.functional.elu(z)
  z=torch.nn.functional.elu(torch.nn.functional.linear(z,self.net[2].weight,self.net[2].bias));z=torch.nn.functional.elu(torch.nn.functional.linear(z,self.net[4].weight,self.net[4].bias));return torch.nn.functional.linear(z,self.net[6].weight,self.net[6].bias)
 def forward(self,x):return self.raw(x).clamp(-1,1)

def fit_dual(world,train,d6pool,hold,walk):
 ih=d22.embed(d22.DirectActor(),"hold").to(world.device).eval();d17.restore_source(world,train,list(range(64)));sobs=world.obs()[:64].detach()
 with torch.inference_mode():sy=hold.mean(sobs).detach()
 basin,bobs,_,_=d22.collect_basin(world,d6pool,walk);basin,bobs=basin.to(world.device),bobs.to(world.device)
 with torch.inference_mode():by=walk(bobs[:,:123],torch.zeros(len(bobs),device=world.device))
 dual=copy.deepcopy(ih);dual.parity_kind=None;timeline,movement=d22.endpoint_fit(dual,sobs.cpu(),sy.cpu(),bobs.cpu(),by.cpu());dual.eval()
 a=Direct143().to(world.device);a.load_state_dict({k:(torch.cat((v,torch.zeros(v.shape[0],2,device=v.device)),1) if k=="net.0.weight" else v) for k,v in dual.state_dict().items()},strict=True);a.parity=True
 x143=torch.cat((sobs,torch.zeros(64,2,device=world.device)),1)
 with torch.inference_mode():parity=f((a.raw(x143)-dual.raw_mean(sobs)).abs().max())
 a.parity=False
 return a.eval(),dual.eval(),sobs,sy,basin,bobs,by,timeline,movement,parity

def mirrored_pool(pool,world,contract):
 perm=torch.tensor(contract["mirror_indices"]);sign=torch.tensor(contract["mirror_signs"],dtype=torch.float32);src={k:v[:32].clone() for k,v in pool["snapshot"].items()};dst={}
 default=world.robot.data.default_joint_pos[0].detach().cpu()
 for k,v in src.items():
  y=v.clone()
  if k=="pose_local":y[:,1]*=-1;y[:,3]*=-1;y[:,5]*=-1
  elif k=="velocity":y[:,1]*=-1;y[:,3]*=-1;y[:,5]*=-1
  elif k=="joint_pos":y=default+sign*(v[:,perm]-default[perm])
  elif k=="joint_vel":y=sign*v[:,perm]
  elif k in ("action","prev_action"):y=sign*v[:,perm]
  elif k in ("physical","previous_physical"):y[:,1:]*=-1
  dst[k]=torch.cat((v,y),0)
 hashes=[]
 for i in range(64):hashes.append(hashlib.sha256(b"".join(dst[k][i].contiguous().numpy().tobytes() for k in sorted(dst))).hexdigest())
 return {"snapshot":dst,"valid":[True]*64,"recipes":list(pool["recipes"][:32])+[f"MIRROR_{x}" for x in pool["recipes"][:32]],"hashes":hashes,"split":"train"}

def action_modes(names,contract,lead="LEFT"):
 idx={n:i for i,n in enumerate(names)};m=torch.zeros(12,37)
 def put(row,vals):
  for n,v in vals.items():m[row,idx[n]]=v
 put(0,{"left_hip_roll_joint":-1,"right_hip_roll_joint":-1});put(1,{"right_hip_roll_joint":1});put(2,{"right_ankle_roll_joint":1});put(3,{"right_knee_joint":-1});put(4,{"left_hip_pitch_joint":1});put(5,{"left_knee_joint":1});put(6,{"left_ankle_pitch_joint":1});put(7,{"left_hip_pitch_joint":1,"right_hip_pitch_joint":1,"left_ankle_pitch_joint":-.5,"right_ankle_pitch_joint":-.5});put(8,{"left_hip_yaw_joint":1,"right_hip_yaw_joint":-1});put(9,{"torso_joint":-1});put(10,{"left_shoulder_pitch_joint":-1,"right_shoulder_pitch_joint":1});put(11,{"left_hip_pitch_joint":.4,"right_hip_pitch_joint":.4,"left_knee_joint":.2,"right_knee_joint":.2,"left_shoulder_pitch_joint":-.2,"right_shoulder_pitch_joint":-.2})
 m=m/m.abs().amax(1,keepdim=True).clamp_min(1e-6)
 if lead=="RIGHT":perm=torch.tensor(contract["mirror_indices"]);sign=torch.tensor(contract["mirror_signs"]);m=sign*m[:,perm]
 return m

def collect_reference(world,pool,walk):
 basin,bobs,by,active=d22.collect_basin(world,pool,walk);forces=[]
 # A separate result-blind steady rollout supplies force thresholds.
 active=active[:64];snap={}
 for k,v in pool["snapshot"].items():
  x=v[active];x=torch.cat((x,x[-1:].expand(world.env.num_envs-len(x),*x.shape[1:])),0);snap[k]=x.to(world.device)
 world.restore_snapshot(snap);rat=[]
 for _ in range(120):
  physical=torch.zeros(world.env.num_envs,3,device=world.device);physical[:,0]=.3;d6.set_command(world,physical);obs124=world.env.observation_manager.compute()["policy"]
  with torch.inference_mode():a=walk(obs124[:,:123],torch.zeros(world.env.num_envs,device=world.device))
  world.wrapped.step(a);force=world.sensor.data.net_forces_w_history[:64,-1,world.sf,:].norm(dim=-1);rat.append(force.cpu())
 force=torch.cat(rat);total=force.sum(1).clamp_min(1e-6);low=force.min(1).values/total;dom=force.max(1).values/total
 # Normalize total support to the reference median, avoiding a mass API dependency.
 med=total.median();return basin.to(world.device),bobs.to(world.device),by.to(world.device),{"swing_low":f(torch.quantile(low,.05)),"swing_high":f(torch.quantile(low,.75)),"support_low":f(torch.quantile(dom,.25)),"support_high":f(torch.quantile(dom,.95)),"total_low":f(torch.quantile(total/med,.05)),"total_high":f(torch.quantile(total/med,.95)),"nominal_force":f(med),"reference_samples":len(force)}

def interpolate_knots(coeff,steps):
 k=coeff.shape[1];t=torch.linspace(0,k-1,steps,device=coeff.device);i=t.floor().long();j=(i+1).clamp(max=k-1);u=(t-i).view(1,steps,1);return coeff[:,i]*(1-u)+coeff[:,j]*u

def sequence(coeff,hold,move,start_basis,pca):
 a=interpolate_knots(coeff[:,:8,:12],40);act_a=(hold[:,None]+.55*torch.einsum("ntk,kj->ntj",a,start_basis)).clamp(-1,1)
 b=interpolate_knots(coeff[:,8:,:8],35);alpha=minimum_jerk(torch.linspace(0,1,35,device=coeff.device)).view(1,35,1);base=act_a[:,-1:,]*(1-alpha)+move[:,None]*alpha;act_b=(base+.35*torch.einsum("ntk,kj->ntj",b,pca)).clamp(-1,1);return torch.cat((act_a,act_b),1)

def evaluate(world,pool,pick,seq,walk,basin,mean,std,thr,lead):
 n=len(seq);d17.restore_source(world,pool,[pick]*n);source,_=d17.nearest_distance(d17.physical_features(world,n),basin,mean,std);flags=[torch.zeros(n,dtype=torch.bool,device=world.device) for _ in range(6)];st=[torch.zeros(n,dtype=torch.long,device=world.device) for _ in range(3)];first=torch.zeros(n,dtype=torch.bool,device=world.device);first_t=torch.full((n,),-1,dtype=torch.long,device=world.device);acq=torch.zeros(n,dtype=torch.bool,device=world.device);acq_t=torch.full_like(first_t,-1);good_streak=torch.zeros(n,dtype=torch.long,device=world.device);flight_streak=torch.zeros(n,dtype=torch.long,device=world.device);total_loss=torch.zeros(n,dtype=torch.bool,device=world.device);yawmax=torch.zeros(n,device=world.device);smooth=torch.zeros(n,device=world.device);prev=world.env.action_manager.prev_action[:n].clone()
 for step in range(100):
  target=d17.set_command(world,step,.5,n);obs=world.obs()
  if step<75:a=seq[:,step]
  else:
   o=world.env.observation_manager.compute()["policy"]
   with torch.inference_mode():a=walk(o[:,:123],torch.zeros(world.env.num_envs,device=world.device))[:n]
  if n<world.env.num_envs:a=torch.cat((a,a[-1:].expand(world.env.num_envs-n,-1)),0)
  _,_,done,extras=world.wrapped.step(a);sf=d15.safety(world,n,done,extras,st)[:6]
  for q,z in zip(flags,sf):q|=z
  smooth+=(a[:n]-prev).square().mean(1);prev=a[:n]
  force=world.sensor.data.net_forces_w_history[:n,-1,world.sf,:].norm(dim=-1);total=force.sum(1);ratio=total/thr["nominal_force"];low=force.min(1).values/total.clamp_min(1e-6);dom=force.max(1).values/total.clamp_min(1e-6);flight=(force<5.).all(1);flight_streak=torch.where(flight,flight_streak+1,torch.zeros_like(flight_streak));total_loss|=flight_streak>=5
  swing=force[:,0 if lead=="LEFT" else 1]/total.clamp_min(1e-6);support=force[:,1 if lead=="LEFT" else 0]/total.clamp_min(1e-6);vel=world.robot.data.root_lin_vel_b[:n];yaw=world.robot.data.root_ang_vel_b[:n,2];yawmax=torch.maximum(yawmax,yaw.abs());grav=world.robot.data.projected_gravity_b[:n];upright=(grav[:,2]<-.85)
  fs=(swing>=thr["swing_low"]-1e-4)&(swing<=thr["swing_high"]+1e-4)&(support>=thr["support_low"]-1e-4)&(support<=thr["support_high"]+1e-4)&(ratio>=thr["total_low"])&(ratio<=thr["total_high"]*1.25)&(yaw.abs()<=.15)&upright&(vel[:,0]>=.03)&(vel[:,0]<=.18)
  new=(~first)&fs&(step<40);first|=new;first_t[new]=step
  good=(vel[:,:2]-target[:,:2]).norm(1)<=.12 if False else ((vel[:,:2]-target[:,:2]).norm(dim=1)<=.12)&(yaw.abs()<=.10)
  good_streak=torch.where(good,good_streak+1,torch.zeros_like(good_streak));newa=(~acq)&(good_streak>=25)&(step<75);acq|=newa;acq_t[newa]=step
 final,_=d17.nearest_distance(d17.physical_features(world,n),basin,mean,std);hard=flags[0]|flags[1]|flags[2]|flags[3]|flags[4]|flags[5]|total_loss;safe=~hard
 return {"safe":safe,"first":first&safe,"acq":acq&safe,"first_t":first_t,"acq_t":acq_t,"basin_ratio":final/source.clamp_min(1e-6),"yaw":yawmax,"smooth":smooth/100,"flags":flags,"total_loss":total_loss}

def cem(world,pool,pick,lead,hold,move,start_basis,pca,walk,basin,mean,std,thr,iterations,candidates,seed):
 torch.manual_seed(seed);dims=15;mu=torch.zeros(1,dims,12,device=world.device);sd=torch.full_like(mu,.35);elite_n=max(2,round(candidates*.1));hist=[];global_score=-float("inf");global_coef=None
 for it in range(iterations):
  chunks=[];outs=[]
  for off in range(0,candidates,world.env.num_envs):
   c=min(world.env.num_envs,candidates-off);coef=(mu+sd*torch.randn(c,dims,12,device=world.device)).clamp(-1,1)
   if off==0:coef[0]=mu[0]
   seq=sequence(coef,hold.expand(c,-1),move.expand(c,-1),start_basis,pca);out=evaluate(world,pool,pick,seq,walk,basin,mean,std,thr,lead);chunks.append(coef);outs.append(out)
  coef=torch.cat(chunks);out={k:(torch.cat([x[k] for x in outs]) if k!="flags" else [torch.cat([x[k][i] for x in outs]) for i in range(6)]) for k in outs[0]}
  # Exact lexicographic ordering: safety > first step > acquisition > basin > yaw > smooth.
  violations=sum(z.float() for z in out["flags"])+out["total_loss"].float();score=torch.where(out["safe"],torch.zeros_like(out["basin_ratio"]),-1e9-violations);score+=out["first"].float()*1e6+out["acq"].float()*1e4-out["basin_ratio"]*100-out["yaw"]-out["smooth"]*.1
  elite=score.topk(elite_n).indices;chosen=coef[elite];mu=chosen.mean(0,keepdim=True);sd=chosen.std(0,keepdim=True).clamp(.03,.5);bi=score.argmax();bs=f(score[bi])
  if bs>global_score:global_score=bs;global_coef=coef[bi:bi+1].detach().clone()
  hist.append({"iteration":it,"safe_candidates":int(out["safe"].sum()),"first_step_candidates":int(out["first"].sum()),"acquisition_candidates":int(out["acq"].sum()),"best_score":bs})
 best_seq=sequence(global_coef,hold,move,start_basis,pca);final=evaluate(world,pool,pick,best_seq,walk,basin,mean,std,thr,lead);return best_seq.detach().cpu(),{"snapshot":pick,"lead":lead,"safe":bool(final["safe"][0]),"first_step":bool(final["first"][0]),"walk_acquisition":bool(final["acq"][0]),"first_step_time":int(final["first_t"][0]),"acquisition_time":int(final["acq_t"][0]),"basin_ratio":f(final["basin_ratio"][0]),"yaw_p95_proxy":f(final["yaw"][0]),"fall":bool(final["flags"][0][0]),"dangerous_slip":bool(final["flags"][1][0]),"impact":bool(final["flags"][2][0]),"velocity_saturation":bool(final["flags"][3][0]),"torque_saturation":bool(final["flags"][4][0]),"total_support_loss":bool(final["total_loss"][0]),"iterations":iterations,"candidates":candidates,"evaluations":iterations*candidates+1,"global_best_score":global_score,"history":hist}

def main():
 parser=argparse.ArgumentParser();add_launcher_args(parser);args,hydra=setup_preset_cli(parser);sys.argv=[sys.argv[0],*hydra];cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=256;cfg.seed=SEED;cfg.episode_length_s=20.;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None
 if args.device:cfg.sim.device=agent.device=args.device
 torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);train=torch.load(D16/"raw/train_start_snapshots.pt",map_location="cpu",weights_only=False);d6pool=torch.load(D6/"raw/snapshots/selected/train_batch_00.pt",map_location="cpu",weights_only=False);contract=json.loads(MIRROR.read_text())
 with launch_simulation(cfg,args):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=d16.StartWorld(wrapped,d3.load_resets(),train);walk=FrozenGaitActor(d3.WMOVE).to(world.device).eval();hold=d3.initialize("P0_STAND_PARENT",world.device)[0].eval();pool=mirrored_pool(train,world,contract)
  actor143,dual,sobs,sy,basin0,bobs,by,timeline,movement,parity=fit_dual(world,train,d6pool,hold,walk);basin,_,by_ref,thr=collect_reference(world,d6pool,walk);basin=basin.to(world.device);by_ref=by_ref.to(world.device);mean,std=basin.mean(0),basin.std(0).clamp_min(1e-4)
  d17.restore_source(world,pool,list(range(64)));source=d17.physical_features(world,64);_,near=d17.nearest_distance(source,basin,mean,std);move=by_ref[near];perm=torch.tensor(contract["mirror_indices"],device=world.device);sign=torch.tensor(contract["mirror_signs"],device=world.device);sy_pool=torch.cat((sy[:32],sign*sy[:32,perm]),0)
  names=contract["joint_names"];left=action_modes(names,contract,"LEFT").to(world.device);right=action_modes(names,contract,"RIGHT").to(world.device);center=by_ref-by_ref.mean(0);_,_,v=torch.pca_lowrank(center,q=8,center=False);pca=v[:,:8].T.contiguous()
  dev=[];seqs=[]
  for pick in list(range(4))+list(range(32,36)):
   for li,lead in enumerate(("LEFT","RIGHT")):
    s,r=cem(world,pool,pick,lead,sy_pool[pick:pick+1],move[pick:pick+1],left if lead=="LEFT" else right,pca,walk,basin,mean,std,thr,12,128,20279711+pick*2+li);dev.append(r);seqs.append(s)
  dev_picks=sorted(set(x["snapshot"] for x in dev));any_first=sum(any(x["snapshot"]==p and x["first_step"] for x in dev) for p in dev_picks)/8;any_acq=sum(any(x["snapshot"]==p and x["walk_acquisition"] for x in dev) for p in dev_picks)/8;dev_gate=any_first>=.75 and any_acq>=.5
  general=[]
  if dev_gate:
   for pick in list(range(4,32))+list(range(36,64)):
    for li,lead in enumerate(("LEFT","RIGHT")):
     s,r=cem(world,pool,pick,lead,sy_pool[pick:pick+1],move[pick:pick+1],left if lead=="LEFT" else right,pca,walk,basin,mean,std,thr,8,64,20279801+pick*2+li);general.append(r);seqs.append(s)
  RAW.mkdir(parents=True,exist_ok=True);np.savez_compressed(RAW/"searched_sequences.npz",**{f"seq_{i:03d}":s.numpy() for i,s in enumerate(seqs)})
  result={"seed":SEED,"parity":parity,"endpoint":{"timeline":timeline,"movement":movement},"mirror_contract_sha":hashlib.sha256(MIRROR.read_bytes()).hexdigest(),"source_hashes":pool["hashes"],"thresholds":thr,"basis":{"left":left.cpu().tolist(),"right":right.cpu().tolist(),"pca_components":8},"development":dev,"development_gate":dev_gate,"generality":general,"temporary_distillation":{"status":"NOT_EXECUTED","reason":"no safe search trajectory" if not any(x["walk_acquisition"] for x in dev+general) else "implementation intentionally deferred to parent after durable oracle dataset assembly"},"persistent_updates":0,"persistent_checkpoints":0,"validation_access":0,"heldout_access":0}
  dump(RAW/"worker_results.json",result);print(json.dumps({"parity":parity,"development_gate":dev_gate,"first":any_first,"acquisition":any_acq,"generality_records":len(general)},indent=2))

if __name__=="__main__":main()
