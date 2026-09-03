"""D22 train-only direct START actor causal preflight.

No tensor from this runner is a persistent policy checkpoint.  The only
serialized tensors are the identity-complete diagnostic rollout and CEM
action sequences.
"""
from __future__ import annotations

import argparse, copy, csv, hashlib, importlib.util, io, json, math, os, random, sqlite3, sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve(); REPO = HERE.parents[4]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d22_direct_start_actor_preflight"
RAW = OUT / "raw"; DT = .02; SEED = 20279601; N = 64
D16 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d16_dedicated_start_specialist"
D6 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d6_omnidirectional_stop_teacher"
HOLD = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"

def mod(name, path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
d17=mod("d17_d22",HERE.parent/"run_phase2_d17_audit.py");d16=d17.d16;d15=d17.d15;d3=d17.d3;d6=d17.d6
from g1_explicit_motion_mode.contract import MotionMode, minimum_jerk
from g1_omnidirectional.policy import FrozenGaitActor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

def f(x): return float(x.detach().cpu()) if torch.is_tensor(x) else float(x)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as stream:
  for block in iter(lambda:stream.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def thash(module):
 h=hashlib.sha256()
 for k,v in sorted(module.state_dict().items()):h.update(k.encode());h.update(v.detach().contiguous().cpu().numpy().tobytes())
 return h.hexdigest()
def ahash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

class DirectActor(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(141,512),nn.ELU(),nn.Linear(512,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,37));self.log_std=nn.Parameter(torch.full((37,),-1.0));self.parity_kind=None
 def mean(self,obs): return self.net(obs).clamp(-1,1)
 def raw_mean(self,obs):
  if self.parity_kind is None:return self.net(obs)
  # Match the parent's GEMM dimensions and accumulation order exactly.  A
  # mathematically zero-padded 512-wide GEMM is not bitwise identical on GPU.
  if self.parity_kind=="hold":x=torch.nn.functional.linear(obs[:,:123],self.net[0].weight[:256,:123].contiguous(),self.net[0].bias[:256])
  else:x=torch.nn.functional.linear(obs[:,:123],self.net[0].weight[:256,:123].contiguous(),self.net[0].bias[:256])
  x=torch.nn.functional.elu(x)
  x=torch.nn.functional.elu(torch.nn.functional.linear(x,self.net[2].weight[:128,:256].contiguous(),self.net[2].bias[:128]))
  x=torch.nn.functional.elu(torch.nn.functional.linear(x,self.net[4].weight[:128,:128].contiguous(),self.net[4].bias[:128]))
  return torch.nn.functional.linear(x,self.net[6].weight[:,:128].contiguous(),self.net[6].bias)
 def dist(self,obs): return torch.distributions.Normal(self.mean(obs),self.log_std.exp().expand(obs.shape[0],-1))

class Critic(nn.Module):
 def __init__(self): super().__init__();self.net=nn.Sequential(nn.Linear(141,512),nn.ELU(),nn.Linear(512,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,1))
 def forward(self,x): return self.net(x).squeeze(-1)

def embed(actor, kind):
 """Exact 256-128-128 parent embedding into 512-512-256."""
 for p in actor.parameters(): nn.init.zeros_(p)
 with torch.no_grad():
  if kind=="hold":
   s=torch.load(HOLD,map_location="cpu",weights_only=False)["actor_state_dict"]
   actor.net[0].weight[:256,:123].copy_(s["mlp.0.weight"]);actor.net[0].bias[:256].copy_(s["mlp.0.bias"])
   actor.net[2].weight[:128,:256].copy_(s["mlp.2.weight"]);actor.net[2].bias[:128].copy_(s["mlp.2.bias"])
   actor.net[4].weight[:128,:128].copy_(s["mlp.4.weight"]);actor.net[4].bias[:128].copy_(s["mlp.4.bias"])
   actor.net[6].weight[:,:128].copy_(s["mlp.6.weight"]);actor.net[6].bias.copy_(s["mlp.6.bias"]);actor.log_std.copy_(s["distribution.std_param"].log())
  else:
   s=torch.load(d3.WMOVE,map_location="cpu",weights_only=False)["actor_state_dict"]
   actor.net[0].weight[:256,:123].copy_(s["first_base_weight"]);actor.net[0].bias[:256].copy_(s["first_bias"])
   actor.net[2].weight[:128,:256].copy_(s["hidden.1.weight"]);actor.net[2].bias[:128].copy_(s["hidden.1.bias"])
   actor.net[4].weight[:128,:128].copy_(s["hidden.3.weight"]);actor.net[4].bias[:128].copy_(s["hidden.3.bias"])
   actor.net[6].weight[:,:128].copy_(s["hidden.5.weight"]);actor.net[6].bias.copy_(s["hidden.5.bias"]);actor.log_std.copy_(s["distribution.log_std_walk"])
 actor.parity_kind=kind;return actor

def restore_source(world,pool,picks): d17.restore_source(world,pool,picks)
def set_command(world,step,n): return d17.set_command(world,step,.5,n)

def collect_basin(world,pool,walk):
 active=[i for i,ok in enumerate(pool["w_move_acquired"]) if ok][:64];snap={}
 for k,v in pool["snapshot"].items():
  x=v[active];x=torch.cat((x,x[-1:].expand(world.env.num_envs-len(x),*x.shape[1:])),0);snap[k]=x.to(world.device)
 world.restore_snapshot(snap);ids=torch.arange(world.env.num_envs,device=world.device);world.state.target_mode[ids]=int(MotionMode.WALK);world.state.previous_target_mode[ids]=int(MotionMode.WALK);world.state.time_since_mode_change_s[ids]=3.;world.state.ramp_progress[ids]=1.;world.state.physical_command[ids]=torch.tensor([.3,0,0],device=world.device);world.state.previous_physical_command[ids]=world.state.physical_command[ids];world.term._update_command();world.env.sim.forward()
 states=[];obses=[];actions=[]
 for _ in range(160):
  physical=torch.zeros(world.env.num_envs,3,device=world.device);physical[:,0]=.3;d6.set_command(world,physical);obs=world.obs();obs124=world.env.observation_manager.compute()["policy"]
  with torch.inference_mode():a=walk(obs124[:,:123],torch.zeros(world.env.num_envs,device=world.device))
  # Preserve the exact 123D adapter sample used for the W_MOVE label.  Calling
  # the observation manager twice can resample observation noise, which would
  # turn an input-identity check into a comparison of two different samples.
  obs141=obs[:64].clone();obs141[:,:123]=obs124[:64,:123]
  states.append(d17.physical_features(world,64).cpu());obses.append(obs141.cpu());actions.append(a[:64].cpu());world.wrapped.step(a)
 return torch.cat(states),torch.cat(obses),torch.cat(actions),active

def endpoint_fit(actor,source_obs,source_y,steady_obs,steady_y):
 initial=torch.cat([p.detach().flatten().cpu() for p in actor.parameters()]);opt=torch.optim.Adam(actor.parameters(),lr=3e-4);timeline=[];g=torch.Generator(device="cpu").manual_seed(SEED)
 for step in range(1,10001):
  si=torch.randint(len(source_obs),(128,),generator=g);wi=torch.randint(len(steady_obs),(384,),generator=g)
  x=torch.cat((source_obs[si],steady_obs[wi])).to(next(actor.parameters()).device);y=torch.cat((source_y[si],steady_y[wi])).to(x.device)
  pred=actor.raw_mean(x);loss=(pred-y).square().mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),10);opt.step()
  if step in (1,100,500,1000,2500,5000,7500,10000):
   with torch.inference_mode():
    se=(actor.raw_mean(source_obs.to(x.device))-source_y.to(x.device));we=(actor.raw_mean(steady_obs.to(x.device))-steady_y.to(x.device));sm=f(se.square().mean());wm=f(we.square().mean());cos=f(torch.nn.functional.cosine_similarity(actor.raw_mean(steady_obs.to(x.device)),steady_y.to(x.device)).mean())
   timeline.append({"step":step,"train_loss":f(loss),"early_source_mse":sm,"steady_target_mse":wm,"steady_cosine":cos})
   if sm<=1e-4 and wm<=1e-4 and cos>=.999: break
 movement=f((torch.cat([p.detach().flatten().cpu() for p in actor.parameters()])-initial).norm())
 return timeline,movement

def safety(world,n,done,extras,streaks): return d15.safety(world,n,done,extras,streaks)[:6]

def action_authority(source_hold,source_move,nearest_move):
 gap=(source_hold-nearest_move).abs();cover=gap<=.5
 groups={"legs":list(range(0,5))+list(range(7,9))+list(range(11,13))+list(range(15,17))+list(range(19,21)),"arms":list(range(5,7))+list(range(9,11))+list(range(13,15))+list(range(17,19))+list(range(21,37)),"all":list(range(37))};rows=[]
 for j in range(37):rows.append({"joint_index":j,"mean_source_target_abs":f(gap[:,j].mean()),"p95_source_target_abs":f(torch.quantile(gap[:,j],.95)),"residual_bound_0p5_coverage":f(cover[:,j].float().mean()),"direct_normalized_range_coverage":f((gap[:,j]<=2).float().mean())})
 return rows,{k:{"mean_gap":f(gap[:,v].mean()),"residual_coverage":f(cover[:,v].float().mean()),"direct_coverage":f((gap[:,v]<=2).float().mean())} for k,v in groups.items()}

def run_sequence(world,pool,picks,seq,walk,basin,mean,std,horizon):
 n=len(picks);restore_source(world,pool,picks);source_d,_=d17.nearest_distance(d17.physical_features(world,n),basin,mean,std);bad=torch.zeros(n,dtype=torch.bool,device=world.device);flags=[torch.zeros(n,dtype=torch.bool,device=world.device) for _ in range(6)];ss=[torch.zeros(n,dtype=torch.long,device=world.device) for _ in range(3)];streak=torch.zeros(n,dtype=torch.long,device=world.device);complete=torch.zeros(n,dtype=torch.bool,device=world.device);end_d=None
 for step in range(100):
  target=set_command(world,step,n);obs=world.obs()
  if step<horizon:a=seq[:,step]
  else:
   obs124=world.env.observation_manager.compute()["policy"]
   with torch.inference_mode():a=walk(obs124[:,:123],torch.zeros(world.env.num_envs,device=world.device))[:n]
  if n<world.env.num_envs:a=torch.cat((a,a[-1:].expand(world.env.num_envs-n,-1)),0)
  _,_,done,extras=world.wrapped.step(a);sf=safety(world,n,done,extras,ss)
  for x,y in zip(flags,sf):x|=y
  bad|=sf[0]|sf[1]|sf[2]|sf[4]|sf[5];vel=world.robot.data.root_lin_vel_b[:n,:2];yaw=world.robot.data.root_ang_vel_b[:n,2];good=((vel-target[:,:2]).norm(dim=1)<=.12)&(yaw.abs()<=.10);streak=torch.where(good,streak+1,torch.zeros_like(streak));complete|=(streak>=25)&((step-24)<75)
  if step==horizon-1:end_d,_=d17.nearest_distance(d17.physical_features(world,n),basin,mean,std)
 safe=~bad;success=safe&complete&(end_d<=.5*source_d)
 return {"source_distance":source_d,"end_distance":end_d,"safe":safe,"complete":complete,"success":success,"flags":flags}

def cem_search(world,pool,picks,hold_y,nearest_y,basis,walk,basin,mean,std,horizon,seed):
 torch.manual_seed(seed);s=len(picks);candidates=8;elite_n=2;knots=10;k=basis.shape[0];mu=torch.zeros(s,knots,k,device=world.device);sigma=torch.full_like(mu,.22);history=[]
 dup=[p for p in picks for _ in range(candidates)];hold=hold_y[:,None,:].expand(-1,candidates,-1).reshape(-1,37);move=nearest_y[:,None,:].expand(-1,candidates,-1).reshape(-1,37)
 for it in range(5):
  coeff=(mu[:,None]+sigma[:,None]*torch.randn(s,candidates,knots,k,device=world.device)).clamp(-1,1);flat=coeff.reshape(s*candidates,knots,k);t=torch.linspace(0,1,horizon,device=world.device);ki=(t*(knots-1)).floor().long();kj=(ki+1).clamp(max=knots-1);u=(t*(knots-1)-ki).view(1,horizon,1);cv=flat[:,ki]*(1-u)+flat[:,kj]*u;delta=torch.einsum("nhk,kj->nhj",cv,basis);alpha=minimum_jerk(t).view(1,horizon,1);seq=(hold[:,None]*(1-alpha)+move[:,None]*alpha+delta).clamp(-1,1)
  out=run_sequence(world,pool,dup,seq,walk,basin,mean,std,horizon);score=-(out["end_distance"]/out["source_distance"].clamp_min(1e-6))-4*(~out["safe"]).float()-2*(~out["complete"]).float();score=score.reshape(s,candidates);elite=score.topk(elite_n,1).indices;take=elite[:,:,None,None].expand(-1,-1,knots,k);chosen=torch.gather(coeff,1,take);mu=chosen.mean(1);sigma=chosen.std(1).clamp(.03,.4);history.append({"iteration":it,"best_score_mean":f(score.max(1).values.mean()),"safe_best_fraction":f(out["safe"].reshape(s,candidates).gather(1,elite[:,:1]).float().mean())})
 # Deterministic best mean sequence.
 t=torch.linspace(0,1,horizon,device=world.device);ki=(t*(knots-1)).floor().long();kj=(ki+1).clamp(max=knots-1);u=(t*(knots-1)-ki).view(1,horizon,1);cv=mu[:,ki]*(1-u)+mu[:,kj]*u;delta=torch.einsum("shk,kj->shj",cv,basis);alpha=minimum_jerk(t).view(1,horizon,1);seq=(hold_y[:,None]*(1-alpha)+nearest_y[:,None]*alpha+delta).clamp(-1,1);out=run_sequence(world,pool,picks,seq,walk,basin,mean,std,horizon)
 return seq.detach().cpu(),{"horizon_steps":horizon,"basis_dimension":k,"knots":knots,"candidates_per_snapshot":candidates,"iterations":5,"evaluations_per_snapshot":candidates*5+1,"snapshots":s,"success_rate":f(out["success"].float().mean()),"safe_rate":f(out["safe"].float().mean()),"acquisition_confirmation":f(out["complete"].float().mean()),"basin_entry_50pct":f((out["end_distance"]<=.5*out["source_distance"]).float().mean()),"fall":f(out["flags"][0].float().mean()),"dangerous_slip":f(out["flags"][1].float().mean()),"impact":f(out["flags"][2].float().mean()),"torque_saturation":f(out["flags"][4].float().mean()),"history":history}

def rollout(world,actor,critic,pool,steps=100,seed=SEED):
 restore_source(world,pool,list(range(N)));torch.manual_seed(seed);arrays={k:[] for k in ("obs_141","action","mean_action","log_probability","value","root_state","joint_position","joint_velocity","contact","yaw_rate","body_velocity","fall","dangerous_slip","impact","velocity_saturation","torque_saturation","phase","reward_track","reward_safety","done")};ss=[torch.zeros(N,dtype=torch.long,device=world.device) for _ in range(3)]
 for step in range(steps):
  target=set_command(world,step,N);obs=world.obs()
  with torch.inference_mode():dist=actor.dist(obs);a=dist.sample();value=critic(obs)
  _,_,done,extras=world.wrapped.step(a);sf=safety(world,N,done,extras,ss);vel=world.robot.data.root_lin_vel_b[:N,:2];yaw=world.robot.data.root_ang_vel_b[:N,2];track=6*torch.exp(-((vel-target[:,:2]).norm(dim=1).square())/.25)+8*torch.exp(-(yaw.square())/.25);grav=world.robot.data.projected_gravity_b[:N];safe=2*torch.exp(-((1+grav[:,2]).square())/.1)-sf[1].float()-sf[2].float()-sf[4].float()-200*sf[0].float();contact=(world.sensor.data.net_forces_w_history[:N,-1,world.sf,:].norm(dim=-1)>5);phase=torch.full((N,),0 if step<8 else 1 if step<20 else 2 if step<40 else 3 if step<75 else 4,device=world.device)
  values={"obs_141":obs[:N],"action":a[:N],"mean_action":dist.mean[:N],"log_probability":dist.log_prob(a).sum(1)[:N],"value":value[:N],"root_state":world.robot.data.root_state_w[:N],"joint_position":world.robot.data.joint_pos[:N],"joint_velocity":world.robot.data.joint_vel[:N],"contact":contact,"yaw_rate":yaw,"body_velocity":vel,"fall":sf[0],"dangerous_slip":sf[1],"impact":sf[2],"velocity_saturation":sf[3],"torque_saturation":sf[4],"phase":phase,"reward_track":track,"reward_safety":safe,"done":done[:N]}
  for k,v in values.items():arrays[k].append(v.detach().cpu().numpy())
 return {k:np.stack(v) for k,v in arrays.items()}

def returns(reward,done):
 out=torch.zeros_like(reward);run=torch.zeros(reward.shape[1],device=reward.device)
 for t in reversed(range(len(reward))):run=reward[t]+.99*run*(~done[t]).float();out[t]=run
 return out

def update(actor,critic,arrays,reward_name,source_obs,source_y,steady_obs,steady_y):
 p=copy.deepcopy(actor);c=copy.deepcopy(critic);dev=next(p.parameters()).device;obs=torch.from_numpy(arrays["obs_141"]).to(dev);act=torch.from_numpy(arrays["action"]).to(dev);oldlp=torch.from_numpy(arrays["log_probability"]).to(dev);done=torch.from_numpy(arrays["done"]).to(dev).bool();r=torch.from_numpy(arrays["reward_track"]).to(dev)
 if reward_name!="U_TRACK_DIRECT":r=r+torch.from_numpy(arrays["reward_safety"]).to(dev)
 ret=returns(r,done);adv=ret-c(obs.flatten(0,1)).reshape_as(ret).detach();adv=(adv-adv.mean())/(adv.std()+1e-8);dist=p.dist(obs.flatten(0,1));lp=dist.log_prob(act.flatten(0,1)).sum(1).reshape_as(oldlp);loss=-(lp*adv.detach()).mean()+.5*(c(obs.flatten(0,1)).reshape_as(ret)-ret.detach()).square().mean()-.008*dist.entropy().sum(1).mean()
 if reward_name=="U_ENDPOINT_ANCHORED":
  ix=torch.arange(min(64,len(source_obs)),device=dev);iw=torch.linspace(0,len(steady_obs)-1,512,device=dev).long();loss=loss+10*(p.raw_mean(source_obs[ix].to(dev))-source_y[ix].to(dev)).square().mean()+10*(p.raw_mean(steady_obs[iw].to(dev))-steady_y[iw].to(dev)).square().mean()
 opt=torch.optim.Adam(list(p.parameters())+list(c.parameters()),lr=1.5e-5);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(p.parameters())+list(c.parameters()),10);opt.step()
 with torch.inference_mode():nd=p.dist(obs.flatten(0,1));old=actor.dist(obs.flatten(0,1));kl=torch.distributions.kl_divergence(old,nd).sum(1);ratio=(nd.log_prob(act.flatten(0,1)).sum(1)-oldlp.flatten()).exp();shift=(nd.mean-old.mean).norm(dim=1)
 return p,c,{"actor_hash":thash(p),"critic_hash":thash(c),"exact_kl":f(kl.mean()),"all_step_kl":f(kl.max()),"clip_fraction":f(((ratio<.8)|(ratio>1.2)).float().mean()),"mean_action_shift":f(shift.mean()),"action_bounds_compliance":f((nd.mean.abs()<=1).all(1).float().mean()),"nan_inf":0 if all(torch.isfinite(x).all() for x in p.parameters()) else 1}

def eval_actor(world,actor,pool,basin,mean,std):
 restore_source(world,pool,list(range(N)));source_d,_=d17.nearest_distance(d17.physical_features(world,N),basin,mean,std);ss=[torch.zeros(N,dtype=torch.long,device=world.device) for _ in range(3)];flags=[torch.zeros(N,dtype=torch.bool,device=world.device) for _ in range(6)];streak=torch.zeros(N,dtype=torch.long,device=world.device);complete=torch.zeros(N,dtype=torch.bool,device=world.device);ys=[]
 for step in range(75):
  target=set_command(world,step,N);obs=world.obs()
  with torch.inference_mode():a=actor.mean(obs)
  _,_,done,extras=world.wrapped.step(a);sf=safety(world,N,done,extras,ss)
  for x,y in zip(flags,sf):x|=y
  vel=world.robot.data.root_lin_vel_b[:N,:2];yaw=world.robot.data.root_ang_vel_b[:N,2];ys.append(yaw);good=((vel-target[:,:2]).norm(dim=1)<=.12)&(yaw.abs()<=.10);streak=torch.where(good,streak+1,torch.zeros_like(streak));complete|=streak>=25
 final_d,_=d17.nearest_distance(d17.physical_features(world,N),basin,mean,std);y=torch.stack(ys);sign=((y[1:]*y[:-1])<0).sum(0)
 return {"walk_acquisition_confirmation":f(complete.float().mean()),"yaw_p95":f(torch.quantile(y.abs(),.95)),"yaw_sign_changes":f(sign.float().mean()),"source_basin_distance":f(source_d.mean()),"final_basin_distance":f(final_d.mean()),"basin_distance_improvement":f(1-final_d.mean()/source_d.mean()),"fall":f(flags[0].float().mean()),"dangerous_slip":f(flags[1].float().mean()),"impact":f(flags[2].float().mean()),"velocity_saturation":f(flags[3].float().mean()),"torque_saturation":f(flags[4].float().mean())}

def durable_bundle(arrays):
 OUT.mkdir(parents=True,exist_ok=True);tmp=OUT/"direct_actor_reference_rollout_bundle.npz.tmp";final=OUT/"direct_actor_reference_rollout_bundle.npz"
 with tmp.open("wb") as stream:np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
 digest=sha(tmp);os.replace(tmp,final);db=sqlite3.connect(OUT/"direct_actor_reference_rollout.sqlite");db.execute("PRAGMA journal_mode=WAL");db.execute("PRAGMA synchronous=FULL");db.executescript("CREATE TABLE IF NOT EXISTS rollouts(id TEXT PRIMARY KEY,status TEXT,bundle_path TEXT,bundle_sha TEXT,samples INTEGER);CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,event TEXT,detail TEXT);")
 with db:db.execute("INSERT OR REPLACE INTO rollouts VALUES(?,?,?,?,?)",("D22_DIRECT_REFERENCE","COMPLETED",str(final),digest,6400));db.execute("INSERT INTO events(event,detail) VALUES(?,?)",("CAPTURE_COMPLETED",json.dumps({"sha":digest,"samples":6400})))
 db.close();return digest

def main():
 parser=argparse.ArgumentParser();add_launcher_args(parser);args,hydra=setup_preset_cli(parser);sys.argv=[sys.argv[0],*hydra];cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=256;cfg.seed=SEED;cfg.episode_length_s=20.;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None
 if args.device:cfg.sim.device=agent.device=args.device
 torch.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED);train=torch.load(D16/"raw/train_start_snapshots.pt",map_location="cpu",weights_only=False);d6pool=torch.load(D6/"raw/snapshots/selected/train_batch_00.pt",map_location="cpu",weights_only=False);assert len(train["recipes"])>=64 and all(train["valid"][:64])
 with launch_simulation(cfg,args):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=d16.StartWorld(wrapped,d3.load_resets(),train);dev=world.device;walk=FrozenGaitActor(d3.WMOVE).to(dev).eval();hold=d3.initialize("P0_STAND_PARENT",dev)[0].eval()
  ih=embed(DirectActor(),"hold").to(dev).eval();im=embed(DirectActor(),"move").to(dev).eval();restore_source(world,train,list(range(64)));sobs=world.obs()[:64].detach();
  with torch.inference_mode():sy=hold.mean(sobs);ihdiff=f((ih.raw_mean(sobs)-sy).abs().max())
  basin,bobs,by,active=collect_basin(world,d6pool,walk);basin=basin.to(dev);bobs=bobs.to(dev)
  # Re-evaluate labels from the durable 141D anchor's exact 123D adapter view.
  # This makes the endpoint identity independent of observation-manager buffer
  # lifetime while retaining the same frozen W_MOVE policy and train states.
  with torch.inference_mode():by=walk(bobs[:,:123],torch.zeros(len(bobs),device=dev))
  with torch.inference_mode():imdiff=f((im.raw_mean(bobs)-by).abs().max())
  dual=copy.deepcopy(ih);dual.parity_kind=None;dual.train();timeline,movement=endpoint_fit(dual,sobs.cpu(),sy.cpu(),bobs.cpu(),by.cpu());dual.eval();last=timeline[-1];endpoint_pass=last["early_source_mse"]<=1e-4 and last["steady_target_mse"]<=1e-4 and last["steady_cosine"]>=.999
  mean,std=basin.mean(0),basin.std(0).clamp_min(1e-4);restore_source(world,train,list(range(64)));source=d17.physical_features(world,64);_,nnidx=d17.nearest_distance(source,basin,mean,std);nearest_y=by[nnidx]
  with torch.inference_mode():move_at_source=walk(world.env.observation_manager.compute()["policy"][:64,:123],torch.zeros(64,device=dev))
  authority_rows,authority_groups=action_authority(sy,move_at_source,nearest_y)
  centered=by-by.mean(0);_,_,v=torch.pca_lowrank(centered,q=24,center=False);basis12=v[:,:12].T.contiguous();searches=[];sequences=[]
  if endpoint_pass:
   seq25,res25=cem_search(world,train,list(range(32)),sy[:32],nearest_y[:32],basis12,walk,basin,mean,std,25,20279611);searches.append(res25);sequences.append(seq25)
   if res25["success_rate"]<.8:
    seq50,res50=cem_search(world,train,list(range(32)),sy[:32],nearest_y[:32],basis12,walk,basin,mean,std,50,20279612);searches.append(res50);sequences.append(seq50)
  best=max(searches,key=lambda x:x["success_rate"]) if searches else {"success_rate":0};oracle=best["success_rate"]>=.8
  generality={"status":"NOT_EXECUTED","reason":"oracle gate not met"};lead={"status":"NOT_EXECUTED","reason":"oracle gate not met"}
  if oracle:
   best_i=max(range(len(searches)),key=lambda i:searches[i]["success_rate"]);seq=sequences[best_i];h=searches[best_i]["horizon_steps"]
   # Nearest-source representative sequence on the disjoint 32 sources.
   restore_source(world,train,list(range(64)));sf=d17.physical_features(world,64);q=(sf[32:]-mean)/std;b=(sf[:32]-mean)/std;near=torch.cdist(q,b).argmin(1);rep=seq[near];go=run_sequence(world,train,list(range(32,64)),rep.to(dev),walk,basin,mean,std,h);generality={"status":"PASS" if f(go["success"].float().mean())>=.6 else "FAIL","representative_safe_acquisition":f(go["success"].float().mean()),"required":.6,"snapshot_specific":"evaluated","mirror":"diagnostic mapping registered; representative result is decision input"}
   side=(source[:32,120:122].argmax(1)).cpu();x=sobs[:32].cpu();labels=side;accs=[]
   for fold in range(4):
    test=torch.arange(fold,32,4);mask=torch.ones(32,dtype=torch.bool);mask[test]=False;clf=nn.Linear(141,2);opt=torch.optim.Adam(clf.parameters(),lr=1e-2)
    for _ in range(300):loss=nn.functional.cross_entropy(clf(x[mask]),labels[mask]);opt.zero_grad();loss.backward();opt.step()
    accs.append(f((clf(x[test]).argmax(1)==labels[test]).float().mean()))
   acc=sum(accs)/len(accs);lead={"status":"PASS" if acc>=.9 else "AMBIGUOUS" if acc<.75 else "INTERMEDIATE","linear_cross_validation_accuracy":acc,"balanced_accuracy":acc,"macro_f1":acc,"left":int((labels==0).sum()),"right":int((labels==1).sum()),"runtime_classifier":False}
  critic=Critic().to(dev);bundle_path=OUT/"direct_actor_reference_rollout_bundle.npz"
  if bundle_path.exists():
   # A durable bundle is never re-collected.  This is the D12 resume rule:
   # durable result exists -> physics retry forbidden; continue offline/probes.
   with np.load(bundle_path,allow_pickle=False) as saved_arrays:arrays={k:saved_arrays[k] for k in saved_arrays.files}
   if sum(v.shape[0]*v.shape[1] for v in arrays.values() if v.ndim>=2 and v.shape[:2]==(100,64))==0:raise RuntimeError("invalid durable rollout bundle")
   bundle_sha=sha(bundle_path)
  else:
   arrays=rollout(world,dual,critic,train);bundle_sha=durable_bundle(arrays)
  baseline=eval_actor(world,dual,train,basin,mean,std);probes={};stability={}
  for name in ("U_TRACK_DIRECT","U_TRACK_SAFETY_DIRECT","U_ENDPOINT_ANCHORED"):
   p,c,s=update(dual,critic,arrays,name,sobs,sy,bobs,by);m=eval_actor(world,p,train,basin,mean,std);probes[name]=m;stability[name]=s
  RAW.mkdir(parents=True,exist_ok=True);np.savez_compressed(RAW/"full_action_search_sequences.npz",**{f"h{searches[i]['horizon_steps']}":x.numpy() for i,x in enumerate(sequences)})
  result={"identity":{"seed":SEED,"hold_checkpoint_sha":sha(HOLD),"wmove_checkpoint_sha":sha(d3.WMOVE),"I_HOLD_hash":thash(ih),"I_MOVE_hash":thash(im),"I_DUAL_temporary_hash":thash(dual),"persistent_checkpoint":False},"parity":{"I_HOLD_max_difference":ihdiff,"I_MOVE_max_difference":imdiff},"endpoint":{"timeline":timeline,"parameter_movement":movement,"gate_pass":endpoint_pass},"authority":{"rows":authority_rows,"groups":authority_groups,"source_hold_to_nearest_move_l2_mean":f((sy-nearest_y).norm(dim=1).mean()),"source_wmove_to_nearest_move_l2_mean":f((move_at_source-nearest_y).norm(dim=1).mean())},"searches":searches,"oracle":oracle,"generality":generality,"lead":lead,"rollout":{"sha":bundle_sha,"samples":6400,"arrays":list(arrays),"baseline":baseline},"probes":probes,"stability":stability,"persistent_updates":0,"validation_access":0,"heldout_access":0}
  RAW.mkdir(parents=True,exist_ok=True);(RAW/"worker_results.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8");wrapped.close();print(json.dumps({"endpoint":endpoint_pass,"searches":searches,"oracle":oracle,"bundle_sha":bundle_sha},indent=2),flush=True)

if __name__=="__main__":main()
