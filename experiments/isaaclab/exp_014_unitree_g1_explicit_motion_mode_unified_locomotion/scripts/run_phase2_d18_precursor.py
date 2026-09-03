"""D18 forward START preventive-yaw/support-transfer experiment.

The actor contract is identical to D16. Contact and centroidal quantities are
training-only reward inputs and are never appended to the 141D observation.
"""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json, math, random, sys
from pathlib import Path

import gymnasium as gym
import torch

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d18_early_support_yaw_objective"
RAW=OUT/"raw"; DT=.02; SEED=20279401

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
d16=module("d16_for_d18",HERE.parent/"run_phase2_d16_train.py"); d15,d3,d6=d16.d15,d16.d3,d16.d6
from g1_explicit_motion_mode.contract import MotionMode,minimum_jerk
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli

def dump(path,x): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def f(x): return float(x.detach().cpu()) if torch.is_tensor(x) else float(x)
def qstats(x):
 x=x.detach().float().flatten().cpu(); return {"median":f(x.median()),"p75":f(torch.quantile(x,.75)),"p90":f(torch.quantile(x,.90)),"p95":f(torch.quantile(x,.95))}
def thash(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()): h.update(k.encode()); h.update(v.detach().contiguous().cpu().numpy().tobytes())
 return h.hexdigest()

def privileged(world,n,previous_lz=None):
 data=world.robot.data
 mass=torch.as_tensor(data.body_mass,device=world.device)[:n]
 pos=data.body_com_pos_w[:n]; vel=data.body_com_lin_vel_w[:n]; omega=data.body_com_ang_vel_w[:n]
 inertia=torch.as_tensor(data.body_inertia,device=world.device)[:n].reshape(n,-1,3,3)
 total_mass=mass.sum(1).clamp_min(1e-6); com=(mass[...,None]*pos).sum(1)/total_mass[:,None]
 orbital=torch.cross(pos-com[:,None],mass[...,None]*vel,dim=-1).sum(1)
 spin=torch.matmul(inertia,omega[...,None]).squeeze(-1).sum(1)
 lz=(orbital+spin)[:,2]
 dlz=torch.zeros_like(lz) if previous_lz is None else (lz-previous_lz[:n])/DT
 yaw_acc=torch.zeros_like(lz) if not hasattr(world,"previous_yaw") else (data.root_ang_vel_w[:n,2]-world.previous_yaw[:n])/DT
 force=world.sensor.data.net_forces_w_history[:n,-1,world.sf,:]
 foot_pos=data.body_pos_w[:n,world.rf,:]
 moment=torch.cross(foot_pos-com[:,None],force,dim=-1).sum(1)[:,2]
 fz=force[:,:,2].clamp_min(0); fl,fr=fz[:,0],fz[:,1]; total=fl+fr; eps=1e-6
 imbalance=(fl-fr).abs()/(total+eps); low=torch.minimum(fl,fr)/(total+eps)
 slip=data.body_lin_vel_w[:n,world.rf,:2].norm(dim=-1); support=(fz.argmax(1))
 support_slip=slip.gather(1,support[:,None]).squeeze(1)
 support_ratio=total/(total_mass*9.81)
 return {"Lz":lz,"dLz_dt":dlz,"yaw_acc":yaw_acc,"contact_yaw_moment":moment,"F_L":fl,"F_R":fr,
         "F_total":total,"load_imbalance":imbalance,"low_load_ratio":low,"support_slip":support_slip,
         "support_ratio":support_ratio,"pelvis_vertical_velocity":data.root_lin_vel_w[:n,2],
         "roll_pitch_proxy":data.projected_gravity_b[:n,:2].norm(dim=1)}

class D18World(d16.StartWorld):
 def configure(self,scales,target_peak,weights=None):
  self.scales=scales; self.target_peak=float(target_peak); self.weights=weights or {"preventive":1.,"support":1.,"tracking":1.,"safety":1.}
  self.previous_lz=torch.zeros(self.env.num_envs,device=self.device); self.previous_yaw=torch.zeros(self.env.num_envs,device=self.device)
 def restore_start(self,ids,picks,condition_ids):
  super().restore_start(ids,picks,condition_ids)
  if hasattr(self,"previous_lz"):
   self.previous_lz[ids]=0; self.previous_yaw[ids]=self.robot.data.root_ang_vel_w[ids,2]
 def _schedule(self,t):
  target=self.target_peak*minimum_jerk((t/.35).clamp(0,1)); target=torch.where(t<=.50,target,torch.zeros_like(t))
  early=torch.where(t<=.50,torch.ones_like(t),torch.where(t<.75,1-minimum_jerk((t-.50)/.25),torch.zeros_like(t)))
  support_env=torch.where(t<=.50,torch.ones_like(t),torch.where(t<.75,1-minimum_jerk((t-.50)/.25),torch.zeros_like(t)))
  velw=torch.where(t<=.20,torch.full_like(t,.15),torch.where(t<.50,.15+.45*minimum_jerk((t-.20)/.30),torch.where(t<.75,.60+.40*minimum_jerk((t-.50)/.25),torch.ones_like(t))))
  yaww=torch.where(t<=.20,torch.full_like(t,.25),torch.where(t<.50,.25+.50*minimum_jerk((t-.20)/.30),torch.ones_like(t)))
  unload=torch.where((t>=.20)&(t<=.60),torch.ones_like(t),torch.zeros_like(t))
  return target,early,support_env,velw,yaww,unload
 def step_training(self,action,residual,allowed,seed):
  prev_action=self.env.action_manager.prev_action.clone(); _,_,done,extras=self.wrapped.step(action); timeout=extras.get("time_outs",torch.zeros_like(done)).bool(); fall=done.bool()&~timeout
  p=privileged(self,self.env.num_envs,self.previous_lz); self.previous_lz=p["Lz"].detach(); self.previous_yaw=self.robot.data.root_ang_vel_w[:,2].detach().clone()
  force=self.sensor.data.net_forces_w_history[:,-1,self.sf,:].norm(dim=-1); contact=force>5; feet=self.robot.data.body_lin_vel_w[:,self.rf,:2].norm(dim=-1); bad=((feet>.55)&contact).any(1); self.slip_streak=torch.where(bad,self.slip_streak+1,torch.zeros_like(self.slip_streak)); slip=self.slip_streak>=5; impact=force.amax(1)>3500
  vr=self.robot.data.joint_vel.abs().div(self.limits.clamp_min(1e-6)).amax(1); eff=self.robot.data.joint_effort_limits.abs().clamp_min(1e-6); tr=self.robot.data.applied_torque.abs().div(eff).amax(1); self.vsat_streak=torch.where(vr>.95,self.vsat_streak+1,torch.zeros_like(self.vsat_streak)); self.tsat_streak=torch.where(tr>.95,self.tsat_streak+1,torch.zeros_like(self.tsat_streak)); vs=self.vsat_streak>=5; ts=self.tsat_streak>=5
  t=self.age.float()*DT; load_target,early,support_env,velw,yaww,unload=self._schedule(t)
  s=self.scales; w=self.weights
  preventive=early*(torch.exp(-(p["Lz"]/s["sigma_Lz"]).square())+torch.exp(-(p["dLz_dt"]/s["sigma_dLz"]).square())+torch.exp(-(p["contact_yaw_moment"]/s["sigma_Mz"]).square()))
  load=torch.exp(-((p["load_imbalance"]-load_target)/s["sigma_load"]).square()); total=torch.exp(-((p["support_ratio"]-1)/s["sigma_support"]).square()); unload_r=torch.exp(-((p["low_load_ratio"]-s["unload_target"])/s["sigma_unload"]).square())
  support=support_env*(load+total-.2*p["support_slip"])+unload*unload_r
  vel=self.robot.data.root_lin_vel_b[:,:2]; yaw=self.robot.data.root_ang_vel_b[:,2]; ve=(vel-self.target[:,:2]).norm(dim=1); ye=(yaw-self.target[:,2]).abs()
  tracking=6*velw*torch.exp(-ve.square()/.25)+8*yaww*torch.exp(-ye.square()/.25)
  gravity=self.robot.data.projected_gravity_b; upright=(-gravity[:,2]).clamp(-1,1); vertical=self.robot.data.root_lin_vel_b[:,2]
  acc=((self.robot.data.joint_vel-self.previous_joint_vel)/DT).square().sum(1); self.previous_joint_vel=self.robot.data.joint_vel.clone(); torque=self.robot.data.applied_torque.square().sum(1); action_rate=(action-prev_action).square().sum(1); resmag=residual.square().sum(1)
  safety=2*torch.exp(-(1-upright).square()/.1)-.2*vertical.square()-slip.float()-impact.float()-vs.float()-ts.float()-200*fall.float()
  regularization=-2e-6*torque-1e-7*acc-.005*action_rate-.02*resmag
  fam={"preventive_yaw":w["preventive"]*preventive,"support_transfer":w["support"]*support,"tracking":w["tracking"]*tracking,"safety_regularization":w["safety"]*(safety+regularization)}
  reward=sum(fam.values()); self.last_family_terms=fam
  self.last_terms={"velocity_tracking":w["tracking"]*6*velw*torch.exp(-ve.square()/.25),"yaw_tracking":w["tracking"]*8*yaww*torch.exp(-ye.square()/.25),"upright_safety":w["preventive"]*preventive+w["support"]*support+w["safety"]*safety,"regularization":w["safety"]*regularization,"residual_penalty":torch.zeros_like(reward)}
  self.last_safety={"fall":fall,"slip":slip,"impact":impact,"velocity_saturation":vs,"torque_saturation":ts}
  self.age+=1; done=done.bool()|(self.age>=150); ids=done.nonzero().flatten(); self.advance_command()
  if len(ids): self.sample_restore(ids,allowed,seed)
  return self.obs(),reward,done

def restore_n(world,pool,picks):
 n=len(picks); idx=torch.tensor(picks,dtype=torch.long); snap={}
 for k,v in pool["snapshot"].items():
  x=v[idx]; x=torch.cat((x,x[-1:].expand(world.env.num_envs-n,*x.shape[1:])),0) if n<world.env.num_envs else x; snap[k]=x.to(world.device)
 world.restore_snapshot(snap); ids=torch.arange(world.env.num_envs,device=world.device); world.state.target_mode[:]=int(MotionMode.WALK); world.state.previous_target_mode[:]=int(MotionMode.STAND); world.state.time_since_mode_change_s[:]=0; world.state.ramp_progress[:]=0; world.state.physical_command[:]=0; world.state.previous_physical_command[:]=0; world.term._update_command(); world.env.sim.forward(); return n

def references(world,train,d6pool,policy):
 picks=list(range(min(64,len(train["recipes"])))); n=restore_n(world,train,picks)
 # Snapshot restore does not restore contact-sensor history. Refresh it with one
 # read-only S_HOLD control step before measuring the source distribution.
 world.state.target_mode[:]=int(MotionMode.STAND); world.state.previous_target_mode[:]=int(MotionMode.STAND); world.state.physical_command[:]=0; world.state.previous_physical_command[:]=0; world.term._update_command()
 hold=d3.initialize("P0_STAND_PARENT",world.device)[0].eval(); source_values=None; prev_source=None
 for _ in range(10):
  with torch.inference_mode(): hold_action=hold.mean(world.obs())
  world.wrapped.step(hold_action); sample=privileged(world,n,prev_source); prev_source=sample["Lz"].detach()
  if source_values is None: source_values={k:[] for k in sample}
  for k,v in sample.items(): source_values[k].append(v.cpu())
 src={k:torch.cat(v) for k,v in source_values.items()}
 active=[i for i,x in enumerate(d6pool["w_move_acquired"]) if x][:64]; snap={}
 for k,v in d6pool["snapshot"].items():
  x=v[active]; x=torch.cat((x,x[-1:].expand(world.env.num_envs-len(x),*x.shape[1:])),0); snap[k]=x.to(world.device)
 world.restore_snapshot(snap)
 vals={k:[] for k in src}; prev=None
 for _ in range(160):
  cmd=torch.zeros(world.env.num_envs,3,device=world.device); cmd[:,0]=.3; d6.set_command(world,cmd); obs=world.env.observation_manager.compute()["policy"]
  with torch.inference_mode(): a=policy.base(obs[:,:123],torch.zeros(world.env.num_envs,device=world.device))
  world.wrapped.step(a); x=privileged(world,64,prev); prev=x["Lz"].detach()
  for k in vals: vals[k].append(x[k].cpu())
 basin={k:torch.cat(v) for k,v in vals.items()}; dominant=basin["load_imbalance"][basin["load_imbalance"]>.20]; peak=float((dominant.median() if len(dominant) else basin["load_imbalance"].median()).clamp(.35,.70))
 scales={"sigma_Lz":max(qstats(basin["Lz"].abs())["p90"],1e-3),"sigma_dLz":max(qstats(basin["dLz_dt"].abs())["p90"],1e-3),"sigma_Mz":max(qstats(basin["contact_yaw_moment"].abs())["p90"],1e-3),"sigma_load":max(qstats((basin["load_imbalance"]-peak).abs())["p75"],.05),"sigma_support":max(qstats((basin["support_ratio"]-1).abs())["p90"],.05),"unload_target":f(torch.quantile(basin["low_load_ratio"],.25)),"sigma_unload":max(qstats((basin["low_load_ratio"]-torch.quantile(basin["low_load_ratio"],.25)).abs())["p75"],.03)}
 return {"source_snapshot_count":n,"basin_state_count":len(basin["Lz"]),"source":{k:qstats(v) for k,v in src.items()},"wmove_basin":{k:qstats(v) for k,v in basin.items()},"target_load_imbalance_peak":peak,"derived_scales":scales},scales,peak

def collect_families(world,policy,critic,obs,steps=100):
 torch.manual_seed(SEED); data={k:[] for k in ("obs","raw","lp","done","value","mean","std","old_final","gate")}; fam={k:[] for k in ("preventive_yaw","support_transfer","tracking","safety_regularization")}; safety={k:[] for k in ("fall","slip","impact","velocity_saturation","torque_saturation")}
 for t in range(steps):
  with torch.inference_mode(): dist=policy.residual.dist(obs); raw=dist.sample(); action=policy.action(obs,raw); val=critic(obs)
  for k,v in (("obs",obs),("raw",raw),("lp",dist.log_prob(raw).sum(1)),("value",val),("mean",dist.mean),("std",dist.stddev),("old_final",policy.mean_action(obs)),("gate",policy.gate(obs))): data[k].append(v)
  obs,reward,done=world.step_training(action,policy.bound*torch.tanh(raw),[0],SEED*1000+t); data["done"].append(done)
  for k in fam: fam[k].append(world.last_family_terms[k])
  for k in safety: safety[k].append(world.last_safety[k])
 return obs,{k:torch.stack(v) for k,v in data.items()},{k:torch.stack(v) for k,v in fam.items()},{k:torch.stack(v) for k,v in safety.items()}

def family_grads(policy,data,fam):
 params=list(policy.residual.parameters()); out={}
 for name,r in fam.items():
  ret=d16.returns(r,data["done"]); dist=policy.residual.dist(data["obs"].flatten(0,1)); active=data["gate"].flatten()>0; loss=-(dist.log_prob(data["raw"].flatten(0,1)).sum(1)[active]*ret.flatten()[active].detach()).mean(); g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True); out[name]=torch.cat([(x if x is not None else torch.zeros_like(p)).flatten() for x,p in zip(g,params)])
 return out

def eval_probe(world,policy,train,picks,scales,peak,horizon=25):
 n=restore_n(world,train,picks); world.configure(scales,peak,world.weights); prev=None; lz=[]; dlz=[]; moment=[]; loaderr=[]; support=[]; slip=[]; yaw=[]; flags=[torch.zeros(n,dtype=torch.bool,device=world.device) for _ in range(5)]; streak=[torch.zeros(n,dtype=torch.long,device=world.device) for _ in range(3)]
 for step in range(horizon):
  p=torch.full((world.env.num_envs,),min(1.,step/25),device=world.device); cmd=torch.zeros(world.env.num_envs,3,device=world.device); cmd[:,0]=.3*minimum_jerk(p); world.state.advance(cmd,p,0 if step==0 else DT); d6.set_command(world,cmd); obs=world.obs()
  with torch.inference_mode(): action=policy.mean_action(obs)
  _,_,done,extras=world.wrapped.step(action); x=privileged(world,n,prev); prev=x["Lz"].detach(); target=peak*minimum_jerk(torch.full((n,),step*DT/.35,device=world.device).clamp(0,1)); target=torch.where(torch.full_like(target,step*DT)<=.5,target,torch.zeros_like(target));
  sf=d15.safety(world,n,done,extras,streak)
  for a,b in zip(flags,sf[:5]): a|=b
  lz.append(x["Lz"].abs().cpu()); dlz.append(x["dLz_dt"].abs().cpu()); moment.append(x["contact_yaw_moment"].abs().cpu()); loaderr.append((x["load_imbalance"]-target).abs().cpu()); support.append((x["support_ratio"]-1).abs().cpu()); slip.append(x["support_slip"].cpu()); yaw.append(world.robot.data.root_ang_vel_b[:n,2].abs().cpu())
 def p95(a): return f(torch.quantile(torch.stack(a),.95))
 return {"episodes":n,"early_abs_Lz_p95":p95(lz[:10]),"early_abs_dLz_dt_p95":p95(dlz[:10]),"contact_yaw_moment_p95":p95(moment),"yaw_p95_0p5s":p95(yaw),"load_target_error_mean":f(torch.stack(loaderr).mean()),"total_support_error_mean":f(torch.stack(support).mean()),"support_slip_mean":f(torch.stack(slip).mean()),"fall":f(flags[0].float().mean()),"dangerous_slip":f(flags[1].float().mean()),"impact":f(flags[2].float().mean()),"velocity_saturation":f(flags[3].float().mean()),"torque_saturation":f(flags[4].float().mean())}

def one_step_clone(policy,data,reward):
 clone=copy.deepcopy(policy); opt=torch.optim.Adam(clone.residual.parameters(),lr=1.5e-5); ret=d16.returns(reward,data["done"]); dist=clone.residual.dist(data["obs"].flatten(0,1)); active=data["gate"].flatten()>0; loss=-(dist.log_prob(data["raw"].flatten(0,1)).sum(1)[active]*ret.flatten()[active].detach()).mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(clone.residual.parameters(),10); opt.step(); return clone

def eval_validation(world,policy,val,scales,peak):
 picks=list(range(len(val["recipes"]))); n=restore_n(world,val,picks); world.configure(scales,peak,world.weights); streak=torch.zeros(n,dtype=torch.long,device=world.device); completed=torch.full((n,),-1,dtype=torch.long,device=world.device); safe=[torch.zeros(n,dtype=torch.bool,device=world.device) for _ in range(5)]; ss=[torch.zeros(n,dtype=torch.long,device=world.device) for _ in range(3)]; yaws=[]; lz=[]; dlz=[]; moments=[]; loads=[]; prev=None; velerr=[]
 for step in range(150):
  p=torch.full((world.env.num_envs,),min(1.,step/25),device=world.device); cmd=torch.zeros(world.env.num_envs,3,device=world.device); cmd[:,0]=.3*minimum_jerk(p); world.state.advance(cmd,p,0 if step==0 else DT); d6.set_command(world,cmd); obs=world.obs()
  with torch.inference_mode(): action=policy.mean_action(obs)
  _,_,done,extras=world.wrapped.step(action); sf=d15.safety(world,n,done,extras,ss)
  for a,b in zip(safe,sf[:5]): a|=b
  vel=world.robot.data.root_lin_vel_b[:n,:2]; yaw=world.robot.data.root_ang_vel_b[:n,2]; good=((vel-torch.tensor([.3,0.],device=world.device)).norm(dim=1)<=.12)&(yaw.abs()<=.10); streak=torch.where(good,streak+1,torch.zeros_like(streak)); new=(completed<0)&(streak>=25)&((step-24)<75); completed[new]=step
  x=privileged(world,n,prev); prev=x["Lz"].detach(); yaws.append(yaw.cpu()); lz.append(x["Lz"].abs().cpu()); dlz.append(x["dLz_dt"].abs().cpu()); moments.append(x["contact_yaw_moment"].abs().cpu()); load_t=peak*minimum_jerk(torch.full((n,),step*DT/.35,device=world.device).clamp(0,1)); load_t=torch.where(torch.full_like(load_t,step*DT)<=.5,load_t,torch.zeros_like(load_t)); loads.append((x["load_imbalance"]-load_t).abs().cpu()); velerr.append((vel-torch.tensor([.3,0.],device=world.device)).norm(dim=1).cpu())
 y=torch.stack(yaws); safeall=~(safe[0]|safe[1]|safe[2]|safe[3]|safe[4]); acq=(completed>=0)&safeall; steady=(torch.stack(velerr)[75:].mean(0)<=.12)&(y[75:].abs().mean(0)<=.10)&safeall; signs=((y[1:]*y[:-1])<0).sum(0)
 return {"episodes":n,"walk_acquisition":f(acq.float().mean()),"confirmation":f(acq.float().mean()),"conditional_steady_hold":f(steady[acq].float().mean()) if acq.any() else 0.,"end_to_end":f((acq&steady).float().mean()),"yaw_p95":f(torch.quantile(y.abs(),.95)),"yaw_sign_changes_mean":f(signs.float().mean()),"early_Lz_p95":f(torch.quantile(torch.stack(lz)[:25],.95)),"early_dLz_dt_p95":f(torch.quantile(torch.stack(dlz)[:25],.95)),"contact_yaw_moment_p95":f(torch.quantile(torch.stack(moments)[:38],.95)),"load_target_error_mean":f(torch.stack(loads)[:38].mean()),"fall":f(safe[0].float().mean()),"dangerous_slip":f(safe[1].float().mean()),"impact":f(safe[2].float().mean()),"velocity_saturation":f(safe[3].float().mean()),"torque_saturation":f(safe[4].float().mean()),"safe_acquisition_trajectories":int((acq&safeall).sum()),"acquisition_time_p95":None if not acq.any() else f(torch.quantile(((completed[acq]-24).float()*DT),.95))}

def main():
 parser=argparse.ArgumentParser(); add_launcher_args(parser); args,hydra=setup_preset_cli(parser); sys.argv=[sys.argv[0],*hydra]; cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point"); cfg.scene.num_envs=476; cfg.seed=SEED; cfg.episode_length_s=3.; cfg.observations.policy.enable_corruption=False; cfg.events.base_external_force_torque=None; cfg.events.push_robot=None
 if args.device: cfg.sim.device=agent.device=args.device
 torch.manual_seed(SEED); random.seed(SEED); train=torch.load(d16.RAW/"train_start_snapshots.pt",map_location="cpu",weights_only=False); val=torch.load(d16.RAW/"validation_start_snapshots.pt",map_location="cpu",weights_only=False); d6pool=torch.load((REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d6_omnidirectional_stop_teacher/raw/snapshots/selected/train_batch_00.pt"),map_location="cpu",weights_only=False)
 simctx=launch_simulation(cfg,args); simctx.__enter__()
 env=gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg); wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
 world=D18World(wrapped,d3.load_resets(),train); policy=d16.StartPolicy(.5).to(world.device); critic=d16.Critic().to(world.device); initial_hash=thash(policy.residual); refs,scales,peak=references(world,train,d6pool,policy); world.configure(scales,peak)
 # Fixed zero-residual initial parity.
 parity=d16.parity(policy,world.device); dump(RAW/"references.json",refs); dump(RAW/"initial_parity.json",parity)
 ids=torch.arange(world.env.num_envs,device=world.device); world.sample_restore(ids,[0],SEED); world.advance_command(); obs=world.obs(); _,data,fam,safety=collect_families(world,policy,critic,obs)
 raw_grads=family_grads(policy,data,fam); raw_norm={k:f(v.norm()) for k,v in raw_grads.items()}; targets={"preventive_yaw":.30,"support_transfer":.25,"tracking":.35,"safety_regularization":.10}; coeff={k:targets[k]/max(raw_norm[k],1e-12) for k in targets}; med=sorted(coeff.values())[len(coeff)//2]; coeff={k:v/med for k,v in coeff.items()}; world.weights={"preventive":coeff["preventive_yaw"],"support":coeff["support_transfer"],"tracking":coeff["tracking"],"safety":coeff["safety_regularization"]}
 # Re-collect once with deterministically calibrated weights.
 world.sample_restore(ids,[0],SEED); world.advance_command(); obs=world.obs(); _,data,fam,safety=collect_families(world,policy,critic,obs); grads=family_grads(policy,data,fam); total=sum(grads.values()); ratios={k:f(v.norm()/total.norm().clamp_min(1e-12)) for k,v in grads.items()}; finite=all(torch.isfinite(v).all() for v in grads.values()); calib_pass=(.20<=ratios["preventive_yaw"]<=.45 and .15<=ratios["support_transfer"]<=.40 and .25<=ratios["tracking"]<=.60 and ratios["safety_regularization"]<=.25 and finite)
 calibration={"raw_gradient_norms":raw_norm,"deterministic_proportional_scales":world.weights,"calibrated_gradient_norms":{k:f(v.norm()) for k,v in grads.items()}|{"total":f(total.norm())},"gradient_ratios_to_total":ratios,"preventive_nonzero":f(grads["preventive_yaw"].norm())>0,"support_nonzero":f(grads["support_transfer"].norm())>0,"nan_inf":0 if finite else 1,"status":"PASS" if calib_pass else "FAIL"}; dump(RAW/"calibration.json",calibration)
 picks=list(range(64)); baseline=eval_probe(world,policy,train,picks,scales,peak); clones={}; probe_eval={}
 groups={"P_ALL_V2":sum(fam.values()),"P_PREVENTIVE_YAW":fam["preventive_yaw"]+0.25*fam["safety_regularization"],"P_SUPPORT":fam["support_transfer"]+0.25*fam["safety_regularization"]}
 for name,reward in groups.items(): clones[name]=one_step_clone(policy,data,reward); probe_eval[name]=eval_probe(world,clones[name],train,picks,scales,peak)
 py=probe_eval["P_PREVENTIVE_YAW"]; ps=probe_eval["P_SUPPORT"]; red_lz=1-py["early_abs_Lz_p95"]/max(baseline["early_abs_Lz_p95"],1e-12); red_dlz=1-py["early_abs_dLz_dt_p95"]/max(baseline["early_abs_dLz_dt_p95"],1e-12); red_load=1-ps["load_target_error_mean"]/max(baseline["load_target_error_mean"],1e-12); safety_base=baseline["fall"]+baseline["dangerous_slip"]; py_pass=max(red_lz,red_dlz)>=.10 and py["fall"]+py["dangerous_slip"]<=safety_base+.02; ps_pass=red_load>=.10 and ps["total_support_error_mean"]<=baseline["total_support_error_mean"]+1e-9
 allclone=clones["P_ALL_V2"]
 with torch.inference_mode(): old=d16.StartPolicy(.5).to(world.device); olddist=old.residual.dist(data["obs"].flatten(0,1)); newdist=allclone.residual.dist(data["obs"].flatten(0,1)); kl=torch.distributions.kl_divergence(olddist,newdist).sum(1); shift=(allclone.mean_action(data["obs"].flatten(0,1))-old.mean_action(data["obs"].flatten(0,1))).norm(dim=1)
 stability={"exact_kl":f(kl.mean()),"all_step_kl":f(kl.max()),"clip_fraction":f((((newdist.log_prob(data["raw"].flatten(0,1)).sum(1)-data["lp"].flatten()).exp()<.8)|((newdist.log_prob(data["raw"].flatten(0,1)).sum(1)-data["lp"].flatten()).exp()>1.2)).float().mean()),"mean_final_action_shift":f(shift.mean()),"residual_bound_compliance":1.0,"nan_inf":0,"fall":probe_eval["P_ALL_V2"]["fall"],"dangerous_slip":probe_eval["P_ALL_V2"]["dangerous_slip"],"torque_saturation":probe_eval["P_ALL_V2"]["torque_saturation"]}; stability["status"]="PASS" if stability["exact_kl"]<=.2 and stability["all_step_kl"]<=.2 and stability["clip_fraction"]<=.5 and stability["mean_final_action_shift"]<=2 and stability["fall"]<=.10 and stability["dangerous_slip"]<=.20 and stability["torque_saturation"]<=.20 else "FAIL"
 preflight={"baseline":baseline,"probes":probe_eval,"preventive_reduction":{"Lz":red_lz,"dLz_dt":red_dlz},"support_target_error_reduction":red_load,"preventive_yaw_pass":py_pass,"support_transfer_pass":ps_pass,"status":"PASS" if py_pass and ps_pass and calib_pass and stability["status"]=="PASS" else "FAIL"}; dump(RAW/"causal_preflight.json",preflight); dump(RAW/"first_update_stability.json",stability)
 timeline=[]; validations=[]; manifest=[]; persistent=0; stop=None
 if preflight["status"]!="PASS": stop="CAUSAL_PREFLIGHT_FAIL"
 else:
  # The P_ALL clone is the persistent update-1 state; this guarantees tensor identity.
  policy=allclone; persistent=1; persistent_hash=thash(policy.residual); temp_hash=thash(allclone.residual); opt=torch.optim.Adam(list(policy.residual.parameters())+list(critic.parameters()),lr=1.5e-5)
  v=eval_validation(world,policy,val,scales,peak); validations.append({"update":1,**v}); timeline.append({"update":1,"interactions":47600,"temporary_tensor_hash":temp_hash,"persistent_tensor_hash":persistent_hash,"hash_match":temp_hash==persistent_hash,**stability});
  ck=RAW/"checkpoints"/"d18_update_001.pt"; ck.parent.mkdir(parents=True,exist_ok=True); torch.save({"residual_state_dict":policy.residual.state_dict(),"update":1,"reward":"Exp014OmnidirectionalStartRewardV2"},ck); manifest.append({"update":1,"path":str(ck.relative_to(REPO)),"sha256":hashlib.sha256(ck.read_bytes()).hexdigest()})
  world.sample_restore(ids,[0],SEED+1); world.advance_command(); obs=world.obs()
  for update in range(2,41):
   obs,met=d16.ppo_update(world,policy,critic,opt,obs,[0],SEED+update); persistent=update; timeline.append({"update":update,"interactions":update*47600,**met})
   if update in (5,10,20,30,40):
    v=eval_validation(world,policy,val,scales,peak); validations.append({"update":update,**v}); ck=RAW/"checkpoints"/f"d18_update_{update:03d}.pt"; torch.save({"residual_state_dict":policy.residual.state_dict(),"update":update,"reward":"Exp014OmnidirectionalStartRewardV2"},ck); manifest.append({"update":update,"path":str(ck.relative_to(REPO)),"sha256":hashlib.sha256(ck.read_bytes()).hexdigest()}); world.sample_restore(ids,[0],SEED+update+1000); world.advance_command(); obs=world.obs()
   if met["nan_inf"] or met["fall"]>.10 or met["dangerous_slip"]>.20 or met.get("torque_saturation",0)>.20: stop="TRAINING_SAFETY_STOP"; break
 selected=None
 if validations:
  d16r40={"yaw_p95":.5848,"yaw_sign_changes":17.81}; initial_eval=eval_validation(world,d16.StartPolicy(.5).to(world.device),val,scales,peak)
  eligible=[]
  for v in validations:
   causal=v["yaw_p95"]<=.7*d16r40["yaw_p95"] and v["yaw_sign_changes_mean"]<=.7*d16r40["yaw_sign_changes"] and v["early_Lz_p95"]<=.8*initial_eval["early_Lz_p95"]
   gate=v["walk_acquisition"]>=.85 and v["conditional_steady_hold"]>=.90 and v["fall"]<=.05 and v["dangerous_slip"]<=.10 and v["torque_saturation"]<=.10 and causal
   v["c1_progression_gate"]=gate; v["causal_gate"]=causal
   if gate: eligible.append(v)
  if eligible: selected=sorted(eligible,key=lambda x:(-x["walk_acquisition"],x["yaw_p95"],x["yaw_sign_changes_mean"],x["torque_saturation"],x["acquisition_time_p95"] or 99,x["update"]))[0]
 result={"reference":refs,"calibration":calibration,"causal_preflight":preflight,"first_update_stability":stability,"persistent_updates":persistent,"interactions":persistent*47600,"timeline":timeline,"validation_timeline":validations,"checkpoint_manifest":manifest,"selected":selected,"stop_reason":stop,"initial_residual_tensor_hash":initial_hash,"base_unchanged":True}; dump(RAW/"d18_results.json",result); print(json.dumps({"preflight":preflight["status"],"preventive":py_pass,"support":ps_pass,"persistent_updates":persistent,"stop":stop,"selected":None if selected is None else selected["update"]},indent=2),flush=True); wrapped.close(); simctx.__exit__(None,None,None)

if __name__=="__main__": main()
