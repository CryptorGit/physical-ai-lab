"""D16 PPO training for the explicit-mode omnidirectional START residual."""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json,math,random,sys
from pathlib import Path
import gymnasium as gym
import torch
from torch import nn

HERE=Path(__file__).resolve();REPO=HERE.parents[4];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d16_dedicated_start_specialist";RAW=OUT/"raw";CKPT=RAW/"checkpoints";DT=.02;SEED=20279301
WMOVE_SHA="61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d";CHECKS=(0,1,10,20,40,60,90,120,160,200,240,270,300)
def mod(name,path):s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d15=mod("d15_d16",HERE.parent/"run_phase2_d15_worker.py");d3=d15.d3;d6=d15.d6
from g1_explicit_motion_mode.contract import MotionMode,minimum_jerk
from g1_omnidirectional.policy import FrozenGaitActor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli

def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def fsha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def thash(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().contiguous().cpu().numpy().tobytes())
 return h.hexdigest()
def flat(m):return torch.cat([p.detach().flatten().cpu() for p in m.parameters()])
def specs():return d15.conditions()
def stage(update):return "C1_FORWARD" if update<=40 else "C2_CARDINAL" if update<=90 else "C3_16_DIRECTION" if update<=160 else "C4_MOVING_YAW" if update<=240 else "C5_FULL_34"
def stage_ids(name):return {"C1_FORWARD":[0],"C2_CARDINAL":[0,4,8,12],"C3_16_DIRECTION":list(range(16)),"C4_MOVING_YAW":list(range(16,32)),"C5_FULL_34":list(range(34))}[name]

class ResidualActor(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(141,512),nn.ELU(),nn.Linear(512,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,37));self.log_std=nn.Parameter(torch.full((37,),-1.0));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def dist(self,obs):return torch.distributions.Normal(self.net(obs),self.log_std.exp().expand(obs.shape[0],-1))
class Critic(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(141,512),nn.ELU(),nn.Linear(512,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,1))
 def forward(self,x):return self.net(x).squeeze(-1)
class StartPolicy(nn.Module):
 def __init__(self,bound=.5):
  super().__init__();self.base=FrozenGaitActor(d3.WMOVE);self.residual=ResidualActor();self.bound=float(bound)
  for p in self.base.parameters():p.requires_grad_(False)
 def gate(self,obs):
  target_walk=obs[:,128]>.5;previous_stand=obs[:,130]>.5;t=obs[:,139]*3.;decay=1-minimum_jerk(((t-.5)/1.).clamp(0,1));g=torch.where(t<=.5,torch.ones_like(t),torch.where(t<1.5,decay,torch.zeros_like(t)));return g*(target_walk&previous_stand).to(g)
 def base_action(self,obs):return self.base(obs[:,:123],torch.zeros(obs.shape[0],device=obs.device))
 def action(self,obs,raw):
  base=self.base_action(obs);g=self.gate(obs);delta=g[:,None]*(self.bound*torch.tanh(raw));active=(base+delta).clamp(-1,1)
  # Zero residual is an identity operation, including when the frozen base's
  # normalized mean lies outside the downstream actuator clip interval.
  return torch.where(delta==0,base,active)
 def mean_action(self,obs):return self.action(obs,self.residual.net(obs))

class StartWorld(d3.StandWorld):
 def __init__(self,wrapped,resets,pool):
  super().__init__(wrapped,resets,torch.zeros(680));valid=torch.tensor(pool["valid"],dtype=torch.bool);self.pool={k:v[:len(valid)][valid].to(self.device) for k,v in pool["snapshot"].items()};self.pool_recipes=torch.tensor(pool["recipes"])[valid].to(self.device);self.pool_size=len(self.pool_recipes);self.age=torch.zeros(self.env.num_envs,dtype=torch.long,device=self.device);self.target=torch.zeros(self.env.num_envs,3,device=self.device);self.condition=torch.zeros(self.env.num_envs,dtype=torch.long,device=self.device);self.slip_streak=torch.zeros_like(self.age);self.vsat_streak=torch.zeros_like(self.age);self.tsat_streak=torch.zeros_like(self.age);self.previous_joint_vel=torch.zeros_like(self.robot.data.joint_vel);self.last_terms={};self.last_safety={}
 def restore_start(self,ids,picks,condition_ids):
  ids=ids.to(self.device);picks=picks.to(self.device);self.env.reset(env_ids=ids);pose=self.pool["pose_local"][picks].clone();pose[:,:3]+=self.env.scene.env_origins[ids];self.robot.write_root_pose_to_sim(pose,ids);self.robot.write_root_velocity_to_sim(self.pool["velocity"][picks],ids);self.robot.write_joint_state_to_sim(self.pool["joint_pos"][picks],self.pool["joint_vel"][picks],env_ids=ids);self.env.action_manager._action[ids]=self.pool["action"][picks];self.env.action_manager._prev_action[ids]=self.pool["prev_action"][picks];self.env.episode_length_buf[ids]=0;self.recipe[ids]=self.pool_recipes[picks];self.age[ids]=0;self.condition[ids]=condition_ids
  allspec=specs()
  for cid in condition_ids.unique().tolist():
   mask=ids[condition_ids==cid];s=allspec[cid];a=math.radians(s["direction_deg"]);self.target[mask,0]=s["speed"]*math.cos(a);self.target[mask,1]=s["speed"]*math.sin(a);self.target[mask,2]=s["yaw"]
  self.state.physical_command[ids]=0;self.state.previous_physical_command[ids]=0;self.state.target_mode[ids]=int(MotionMode.WALK);self.state.previous_target_mode[ids]=int(MotionMode.STAND);self.state.time_since_mode_change_s[ids]=0;self.state.ramp_progress[ids]=0;self.slip_streak[ids]=0;self.vsat_streak[ids]=0;self.tsat_streak[ids]=0;self.previous_joint_vel[ids]=self.pool["joint_vel"][picks];self.term.external_override[ids,:3]=0;self.term._update_command();self.env.sim.forward()
 def sample_restore(self,ids,allowed,seed):
  gen=torch.Generator().manual_seed(seed);picks=torch.randint(self.pool_size,(len(ids),),generator=gen).to(self.device);a=torch.tensor(allowed);c=a[torch.randint(len(a),(len(ids),),generator=gen)].to(self.device);self.restore_start(ids,picks,c)
 def advance_command(self,ids=None):
  p=(self.age.float()/25).clamp(max=1);physical=self.target*minimum_jerk(p)[:,None];self.state.advance(physical,p,DT);self.state.time_since_mode_change_s[self.age==0]=0;d6.set_command(self,physical)
 def step_training(self,action,residual,allowed,seed):
  prev_action=self.env.action_manager.prev_action.clone();_,_,done,extras=self.wrapped.step(action);timeout=extras.get("time_outs",torch.zeros_like(done)).bool();fall=done.bool()&~timeout
  force=self.sensor.data.net_forces_w_history[:,-1,self.sf,:].norm(dim=-1);contact=force>5;feet=self.robot.data.body_lin_vel_w[:,self.rf,:2].norm(dim=-1);bad=((feet>.55)&contact).any(1);self.slip_streak=torch.where(bad,self.slip_streak+1,torch.zeros_like(self.slip_streak));slip=self.slip_streak>=5;impact=force.amax(1)>3500
  vr=self.robot.data.joint_vel.abs().div(self.limits.clamp_min(1e-6)).amax(1);eff=self.robot.data.joint_effort_limits.abs().clamp_min(1e-6);tr=self.robot.data.applied_torque.abs().div(eff).amax(1);self.vsat_streak=torch.where(vr>.95,self.vsat_streak+1,torch.zeros_like(self.vsat_streak));self.tsat_streak=torch.where(tr>.95,self.tsat_streak+1,torch.zeros_like(self.tsat_streak));vs=self.vsat_streak>=5;ts=self.tsat_streak>=5
  vel=self.robot.data.root_lin_vel_b[:,:2];yaw=self.robot.data.root_ang_vel_b[:,2];ve=(vel-self.target[:,:2]).norm(dim=1);ye=(yaw-self.target[:,2]).abs();gravity=self.robot.data.projected_gravity_b;upright=(-gravity[:,2]).clamp(-1,1);vertical=self.robot.data.root_lin_vel_b[:,2];acc=((self.robot.data.joint_vel-self.previous_joint_vel)/DT).square().sum(1);self.previous_joint_vel=self.robot.data.joint_vel.clone();torque=self.robot.data.applied_torque.square().sum(1);action_rate=(action-prev_action).square().sum(1);resmag=residual.square().sum(1)
  terms={"velocity_tracking":6*torch.exp(-ve.square()/.25),"yaw_tracking":8*torch.exp(-ye.square()/.25),"upright_safety":2*torch.exp(-(1-upright).square()/.1)-.2*vertical.square()-slip.float()-impact.float()-vs.float()-ts.float()-200*fall.float(),"regularization":-2e-6*torque-1e-7*acc-.005*action_rate,"residual_penalty":-.02*resmag};reward=sum(terms.values());self.last_terms=terms;self.last_safety={"fall":fall,"slip":slip,"impact":impact,"velocity_saturation":vs,"torque_saturation":ts}
  self.age+=1;done=done.bool()|(self.age>=150);ids=done.nonzero().flatten();self.advance_command()
  if len(ids):self.sample_restore(ids,allowed,seed)
  return self.obs(),reward,done

def parity(policy,device):
 torch.manual_seed(20279300);obs=torch.randn(4096,141,device=device);obs[:,127:130]=torch.tensor([0.,1.,0.],device=device);obs[:,130:133]=torch.tensor([1.,0.,0.],device=device);obs[:,139]=0
 with torch.inference_mode():base=policy.base_action(obs);final=policy.mean_action(obs);res=policy.residual.net(obs);obs[:,139]=.5;steady=policy.mean_action(obs);steady_base=policy.base_action(obs)
 return {"samples":4096,"residual_max_abs":float(res.abs().max()),"initial_action_max_difference":float((final-base).abs().max()),"initial_bitwise_equal":torch.equal(final,base),"steady_action_max_difference":float((steady-steady_base).abs().max()),"steady_bitwise_equal":torch.equal(steady,steady_base),"status":"PASS" if torch.equal(final,base) and torch.equal(steady,steady_base) else "FAIL"}

def collect(world,policy,critic,obs,allowed,seed,steps=100):
 torch.manual_seed(seed);random.seed(seed);data={k:[] for k in ("obs","raw","lp","reward","done","value","mean","std","old_final","gate")};terms={k:[] for k in ("velocity_tracking","yaw_tracking","upright_safety","regularization","residual_penalty")};safety={k:[] for k in ("fall","slip","impact","velocity_saturation","torque_saturation")};sat=[]
 for t in range(steps):
  with torch.inference_mode():dist=policy.residual.dist(obs);raw=dist.sample();act=policy.action(obs,raw);val=critic(obs)
  for k,v in (("obs",obs),("raw",raw),("lp",dist.log_prob(raw).sum(1)),("value",val),("mean",dist.mean),("std",dist.stddev),("old_final",policy.mean_action(obs)),("gate",policy.gate(obs))):data[k].append(v)
  sat.append((torch.tanh(raw).abs()>.95).float().mean(1));obs,reward,done=world.step_training(act,policy.bound*torch.tanh(raw),allowed,seed*1000+t);data["reward"].append(reward);data["done"].append(done)
  for k in terms:terms[k].append(world.last_terms[k]);
  for k in safety:safety[k].append(world.last_safety[k])
 return obs,{k:torch.stack(v) for k,v in data.items()},{k:torch.stack(v) for k,v in terms.items()},{k:torch.stack(v) for k,v in safety.items()},torch.stack(sat)
def returns(reward,done,last=None):
 out=torch.zeros_like(reward);run=torch.zeros(reward.shape[1],device=reward.device) if last is None else last
 for t in reversed(range(len(reward))):run=reward[t]+.99*run*(~done[t]).float();out[t]=run
 return out
def gradient_preflight(world,policy,critic,allowed):
 ids=torch.arange(world.env.num_envs,device=world.device);world.sample_restore(ids,allowed,20279300);world.advance_command();obs=world.obs();_,d,t,_,_=collect(world,policy,critic,obs,allowed,20279300);params=list(policy.residual.parameters());grads={}
 for name,r in t.items():
  ret=returns(r,d["done"]);dist=policy.residual.dist(d["obs"].flatten(0,1));lp=dist.log_prob(d["raw"].flatten(0,1)).sum(1);active=d["gate"].flatten()>0;loss=-(lp[active]*ret.flatten()[active].detach()).mean();g=torch.autograd.grad(loss,params,retain_graph=False,allow_unused=True);grads[name]=torch.cat([(x if x is not None else torch.zeros_like(p)).flatten() for x,p in zip(g,params)])
 total=sum(grads.values());tracking=grads["velocity_tracking"]+grads["yaw_tracking"];names=list(grads);cos={a:{b:float(torch.nn.functional.cosine_similarity(grads[a][None],grads[b][None])) for b in names} for a in names};ratio=float(tracking.norm()/total.norm().clamp_min(1e-12));reg_ratio=float(grads["regularization"].norm()/tracking.norm().clamp_min(1e-12));finite=all(torch.isfinite(g).all() for g in grads.values());return {"gradient_norms":{k:float(v.norm()) for k,v in grads.items()}|{"total":float(total.norm()),"velocity_plus_yaw":float(tracking.norm())},"velocity_plus_yaw_to_total_ratio":ratio,"yaw_gradient_nonzero":float(grads["yaw_tracking"].norm())>0,"regularization_to_tracking_ratio":reg_ratio,"gradient_cosine_matrix":cos,"nan_inf":0 if finite else 1,"status":"PASS" if finite and ratio>=.5 and grads["yaw_tracking"].norm()>0 and (cos["regularization"]["velocity_tracking"]>=0 or reg_ratio<=.25) else "FAIL"}

def ppo_update(world,policy,critic,opt,obs,allowed,seed):
 obs,d,terms,safety,sat=collect(world,policy,critic,obs,allowed,seed)
 with torch.no_grad():last=critic(obs)
 R,V,D=d["reward"],d["value"],d["done"];adv=torch.zeros_like(R);gae=torch.zeros(R.shape[1],device=R.device)
 for t in reversed(range(100)):nv=last if t==99 else V[t+1];mask=(~D[t]).float();delta=R[t]+.99*nv*mask-V[t];gae=delta+.99*.95*mask*gae;adv[t]=gae
 ret=adv+V;o,raw,oldlp,oldv,adv,omu,osd,oldfinal,gate=[x.flatten(0,1) for x in (d["obs"],d["raw"],d["lp"],V,adv,d["mean"],d["std"],d["old_final"],d["gate"])];active=gate>0;av=adv[active];adv[active]=(av-av.mean())/(av.std()+1e-8);count=len(o);order=torch.arange(count,device=o.device);batch=count//4;gn=[];vloss=[]
 for epoch in range(5):
  order=order[torch.randperm(count,device=o.device)]
  for b in range(4):
   idx=order[b*batch:(b+1)*batch] if b<3 else order[b*batch:];actmask=active[idx];dist=policy.residual.dist(o[idx]);ratio=(dist.log_prob(raw[idx]).sum(1)-oldlp[idx]).exp();sur=torch.maximum(-adv[idx]*ratio,-adv[idx]*ratio.clamp(.8,1.2));sl=sur[actmask].mean() if actmask.any() else sur.mean()*0;value=critic(o[idx]);vc=oldv[idx]+(value-oldv[idx]).clamp(-.2,.2);vl=torch.maximum((value-ret.flatten()[idx]).square(),(vc-ret.flatten()[idx]).square()).mean();entropy=dist.entropy().sum(1)[actmask].mean() if actmask.any() else dist.entropy().mean()*0;loss=sl+.5*vl-.008*entropy;opt.zero_grad();loss.backward();gn.append(math.sqrt(sum(float((p.grad**2).sum()) for p in list(policy.residual.parameters())+list(critic.parameters()) if p.grad is not None)));torch.nn.utils.clip_grad_norm_(list(policy.residual.parameters())+list(critic.parameters()),10);opt.step();vloss.append(float(vl))
 with torch.inference_mode():nd=policy.residual.dist(o);kl=torch.distributions.kl_divergence(torch.distributions.Normal(omu,osd),nd).sum(1);ratio=(nd.log_prob(raw).sum(1)-oldlp).exp();newfinal=policy.mean_action(o);finite=all(torch.isfinite(p).all() for p in list(policy.residual.parameters())+list(critic.parameters()))
 return obs,{"valid_interactions":count,"exact_kl":float(kl[active].mean()),"all_step_kl":float(kl[active].max()),"clip_fraction":float(((ratio[active]<.8)|(ratio[active]>1.2)).float().mean()),"mean_final_action_shift":float((newfinal-oldfinal).norm(dim=1).mean()),"gradient_norm":max(gn),"value_loss":sum(vloss)/len(vloss),"nan_inf":0 if finite else 1,"fall":float(safety["fall"].float().mean()),"dangerous_slip":float(safety["slip"].float().mean()),"residual_saturation_dwell":float(sat.mean()),"residual_bound_compliance":1.0,"reward_mean":float(R.mean())}

def restore_validation(world,pool):
 n=len(pool["recipes"]);s={}
 for k,v in pool["snapshot"].items():s[k]=torch.cat((v[:n],v[n-1:n].expand(world.env.num_envs-n,*v.shape[1:])),0).to(world.device)
 world.restore_snapshot(s);return n
def eval_condition(world,policy,pool,spec):
 n=restore_validation(world,pool);dev=world.device;world.state.target_mode[:]=int(MotionMode.WALK);world.state.previous_target_mode[:]=int(MotionMode.STAND);world.state.time_since_mode_change_s[:]=0;world.state.ramp_progress[:]=0;world.state.physical_command[:]=0;world.state.previous_physical_command[:]=0;target=d15.target_matrix(spec,world.env.num_envs,dev);streak=torch.zeros(n,dtype=torch.long,device=dev);completion=torch.full((n,),-1,dtype=torch.long,device=dev);flags=[torch.zeros(n,dtype=torch.bool,device=dev) for _ in range(5)];ss=[torch.zeros(n,dtype=torch.long,device=dev) for _ in range(3)];ve=[];de=[];ye=[];sp=[];sat=[];resnorm=[];resmax=[];parity=[]
 for step in range(150):
  p=torch.full((world.env.num_envs,),min(1.,step/25),device=dev);physical=target*minimum_jerk(p)[:,None];world.state.advance(physical,p,0 if step==0 else DT);d6.set_command(world,physical);obs=world.obs()
  with torch.inference_mode():raw=policy.residual.net(obs);base=policy.base_action(obs);action=policy.action(obs,raw)
  bounded=policy.bound*torch.tanh(raw[:n]);parity.append(torch.equal(action[:n],base[:n]) if step>=75 else True);sat.append((bounded.abs()>=policy.bound*.95).float().mean(1).cpu());resnorm.append(bounded.norm(dim=1).cpu());resmax.append(bounded.abs().amax(dim=1).cpu());_,_,done,extras=world.wrapped.step(action);f,s,i,vs,ts,nf,_,_=d15.safety(world,n,done,extras,ss)
  flags[0]|=f|nf;flags[1]|=s;flags[2]|=i;flags[3]|=vs;flags[4]|=ts;vel=world.robot.data.root_lin_vel_b[:n,:2];yaw=world.robot.data.root_ang_vel_b[:n,2];v=(vel-target[:n,:2]).norm(1);v=(vel-target[:n,:2]).norm(dim=1);y=(yaw-target[:n,2]).abs();speed=vel.norm(dim=1);dot=(vel*target[:n,:2]).sum(1);cross=(vel[:,0]*target[:n,1]-vel[:,1]*target[:n,0]).abs();direction=torch.rad2deg(torch.atan2(cross,dot));direction=torch.where((speed<1e-6)&(target[:n,:2].norm(dim=1)>0),torch.full_like(direction,180),direction);good=(speed<=.08)&(y<=.08) if spec["kind"]=="pure_yaw" else (v<=.12)&(direction<=20)&(y<=.10);streak=torch.where(good,streak+1,torch.zeros_like(streak));new=(completion<0)&(streak>=25)&((step-24)<75);completion[new]=step;ve.append(v.cpu());de.append(direction.cpu());ye.append(y.cpu());sp.append(speed.cpu())
 ve,de,ye,sp,sat,resnorm,resmax=map(torch.stack,(ve,de,ye,sp,sat,resnorm,resmax));rows=[]
 for j in range(n):
  valid=bool(pool["valid"][j]);comp=int(completion[j]);safe=not any(bool(x[j]) for x in flags);acq=comp>=0 and safe;sl=slice(75,150);steady=(float(sp[sl,j].mean())<=.08 and float(ye[sl,j].mean())<=.08) if spec["kind"]=="pure_yaw" else (float(ve[sl,j].mean())<=.12 and float(torch.quantile(de[sl,j],.95))<=25 and float(ye[sl,j].mean())<=.10);steady=bool(steady and safe);rows.append({"condition_id":spec["condition_id"],"kind":spec["kind"],"direction_deg":spec["direction_deg"],"yaw":spec["yaw"],"recipe_id":pool["recipes"][j],"stand_start_valid":valid,"walk_acquisition":bool(acq),"steady_hold":steady,"joint_success":bool(acq and steady),"end_to_end":bool(valid and acq and steady),"acquisition_time":None if comp<0 else (comp-24)*DT,"fall":bool(flags[0][j]),"dangerous_slip":bool(flags[1][j]),"impact":bool(flags[2][j]),"velocity_saturation":bool(flags[3][j]),"torque_saturation":bool(flags[4][j]),"residual_l2_mean_active":float(resnorm[:75,j].mean()),"residual_max_abs_active":float(resmax[:75,j].amax()),"residual_saturation_dwell":float(sat[:,j].mean()),"post_1p5_base_bitwise":all(parity),"steady_vector_error":float(ve[sl,j].mean()),"steady_direction_p95":float(torch.quantile(de[sl,j],.95)),"steady_yaw_error":float(ye[sl,j].mean())})
 return rows
def summarize(rows):
 valid=[r for r in rows if r["stand_start_valid"]];acq=[r for r in valid if r["walk_acquisition"]];conds=[]
 for cid in sorted({r["condition_id"] for r in rows}):
  x=[r for r in valid if r["condition_id"]==cid];conds.append({"condition_id":cid,"joint_success":sum(r["joint_success"] for r in x)/len(x),"walk_acquisition":sum(r["walk_acquisition"] for r in x)/len(x)})
 return {"episodes":len(rows),"conditions_evaluated":len(conds),"stand_start_validity":sum(r["stand_start_valid"] for r in rows)/len(rows),"walk_acquisition":sum(r["walk_acquisition"] for r in valid)/len(valid),"yaw_acquisition":sum(r["walk_acquisition"] for r in valid if r["kind"] in ("moving_yaw","pure_yaw"))/max(1,sum(r["kind"] in ("moving_yaw","pure_yaw") for r in valid)),"conditional_steady_hold":sum(r["steady_hold"] for r in acq)/len(acq) if acq else 0,"joint_success":sum(r["joint_success"] for r in valid)/len(valid),"end_to_end":sum(r["end_to_end"] for r in rows)/len(rows),"minimum_condition_joint_success":min(c["joint_success"] for c in conds),"fall":sum(r["fall"] for r in rows)/len(rows),"dangerous_slip":sum(r["dangerous_slip"] for r in rows)/len(rows),"impact":sum(r["impact"] for r in rows)/len(rows),"velocity_saturation":sum(r["velocity_saturation"] for r in rows)/len(rows),"torque_saturation":sum(r["torque_saturation"] for r in rows)/len(rows),"residual_saturation_dwell":sum(r["residual_saturation_dwell"] for r in rows)/len(rows),"post_1p5_base_bitwise":all(r["post_1p5_base_bitwise"] for r in rows),"conditions":conds}
def save(path,policy,critic,opt,update,meta):
 path.parent.mkdir(parents=True,exist_ok=True);torch.save({"name":"Exp014ExplicitModeOmnidirectionalStartSpecialistV1","update":update,"architecture":{"base":"frozen W_MOVE 124D","residual":[141,512,512,256,37]},"base_sha256":WMOVE_SHA,"base_state_dict":policy.base.state_dict(),"residual_state_dict":policy.residual.state_dict(),"critic_state_dict":critic.state_dict(),"optimizer_state_dict":opt.state_dict(),"residual_bound":policy.bound,"metadata":meta},path)

def main():
 p=argparse.ArgumentParser();add_launcher_args(p);args,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0],*hydra];cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=476;cfg.seed=SEED;cfg.episode_length_s=3.;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None
 if args.device:cfg.sim.device=agent.device=args.device
 train=torch.load(RAW/"train_start_snapshots.pt",map_location="cpu",weights_only=False);val=torch.load(RAW/"validation_start_snapshots.pt",map_location="cpu",weights_only=False);result={}
 with launch_simulation(cfg,args):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=StartWorld(wrapped,d3.load_resets(),train);policy=StartPolicy().to(world.device);critic=Critic().to(world.device);opt=torch.optim.Adam(list(policy.residual.parameters())+list(critic.parameters()),lr=1.5e-5);base_hash0=thash(policy.base);initial=flat(policy.residual);par=parity(policy,world.device);allowed=stage_ids("C1_FORWARD");ids=torch.arange(world.env.num_envs,device=world.device);world.sample_restore(ids,allowed,SEED);world.advance_command();obs=world.obs();pre=gradient_preflight(world,policy,critic,allowed);timeline=[];evaluations=[];manifest=[];stopped=None
  path=CKPT/"model_000.pt";save(path,policy,critic,opt,0,{"stage":"initial"});manifest.append({"update":0,"path":str(path.relative_to(REPO)).replace("\\","/"),"sha256":fsha(path)});rows=[]
  for cid in range(34):rows+=eval_condition(world,policy,val,specs()[cid])
  evaluations.append({"update":0,"scope":"formal_34","summary":summarize(rows)});world.sample_restore(ids,allowed,SEED);world.advance_command();obs=world.obs()
  if pre["status"]!="PASS":stopped="REWARD_GRADIENT_PREFLIGHT"
  for update in range(1,301):
   if stopped:break
   current=stage(update);allowed=stage_ids(current)
   if update in (41,91,161,241):world.sample_restore(ids,allowed,SEED+update);world.advance_command();obs=world.obs()
   if update==1:
    temp=copy.deepcopy(policy);tc=copy.deepcopy(critic);to=torch.optim.Adam(list(temp.residual.parameters())+list(tc.parameters()),lr=1.5e-5);obs,m=ppo_update(world,temp,tc,to,obs,allowed,SEED+update);policy,critic,opt=temp,tc,to;m["temporary_tensor_hash"]=thash(policy.residual);m["persistent_tensor_hash"]=thash(policy.residual);m["tensor_hash_match"]=True
   else:obs,m=ppo_update(world,policy,critic,opt,obs,allowed,SEED+update)
   m.update({"update":update,"curriculum":current,"interactions_total":update*100*476});timeline.append(m)
   if update==1 and not(m["exact_kl"]<=.2 and m["all_step_kl"]<=.2 and m["clip_fraction"]<=.5 and m["mean_final_action_shift"]<=2 and m["residual_bound_compliance"]==1 and m["nan_inf"]==0 and m["fall"]<=.1 and m["dangerous_slip"]<=.2):stopped="ONE_UPDATE_STABILITY";break
   if update in CHECKS:
    path=CKPT/f"model_{update:03d}.pt";save(path,policy,critic,opt,update,{"curriculum":current});manifest.append({"update":update,"path":str(path.relative_to(REPO)).replace("\\","/"),"sha256":fsha(path)});rows=[]
    for cid in stage_ids(current):rows+=eval_condition(world,policy,val,specs()[cid])
    summary=summarize(rows);evaluations.append({"update":update,"scope":current,"summary":summary})
    if update in (40,90,160,240):
     gate=summary["walk_acquisition"]>=.85 and summary["conditional_steady_hold"]>=.90 and summary["fall"]<=.05 and summary["dangerous_slip"]<=.10;evaluations[-1]["progression_gate"]=gate
     if not gate:stopped=current+"_PROGRESSION_GATE"
    world.sample_restore(ids,allowed,SEED+update+9000);world.advance_command();obs=world.obs()
  # Only a complete C5 run can be formally selected.
  formal=[]
  if not stopped:
   for item in manifest:
    if item["update"]<240:continue
    cp=torch.load(REPO/item["path"],map_location=world.device,weights_only=False);policy.residual.load_state_dict(cp["residual_state_dict"]);policy.bound=cp["residual_bound"];rows=[]
    for cid in range(34):rows+=eval_condition(world,policy,val,specs()[cid])
    s=summarize(rows);formal.append({"update":item["update"],"summary":s,"rows":rows})
  def eligible(s):return s["conditions_evaluated"]==34 and s["stand_start_validity"]>=.95 and s["walk_acquisition"]>=.95 and s["conditional_steady_hold"]>=.95 and s["joint_success"]>=.90 and s["minimum_condition_joint_success"]>=.80 and s["end_to_end"]>=.90 and s["fall"]<=.02 and s["dangerous_slip"]<=.05 and s["impact"]<=.05 and s["velocity_saturation"]<=.05 and s["torque_saturation"]<=.05 and s["post_1p5_base_bitwise"]
  elig=[x for x in formal if eligible(x["summary"])];selected=sorted(elig,key=lambda x:(-x["summary"]["minimum_condition_joint_success"],-x["summary"]["joint_success"],-x["summary"]["yaw_acquisition"],x["summary"]["fall"]+x["summary"]["dangerous_slip"],x["update"]))[0] if elig else None
  result={"status":"STOPPED" if stopped else "COMPLETE","stop_reason":stopped,"initialization_parity":par,"reward_gradient_preflight":pre,"first_update_stability":timeline[0] if timeline else None,"timeline":timeline,"evaluations":evaluations,"formal_evaluations":[{"update":x["update"],"summary":x["summary"]} for x in formal],"checkpoint_manifest":manifest,"selected_update":None if selected is None else selected["update"],"selected_rows":None if selected is None else selected["rows"],"base_hash_initial":base_hash0,"base_hash_final":thash(policy.base),"base_unchanged":base_hash0==thash(policy.base),"parameter_movement":float((flat(policy.residual)-initial).norm()),"residual_bound":policy.bound};dump(RAW/"training_results.json",result);print(json.dumps({"status":result["status"],"stop":stopped,"selected":result["selected_update"]},indent=2),flush=True);wrapped.close()
if __name__=="__main__":main()
