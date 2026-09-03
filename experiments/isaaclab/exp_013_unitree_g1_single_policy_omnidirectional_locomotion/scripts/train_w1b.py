"""Strict-resume preflight and single authorized W1B PPO continuation."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
import gymnasium as gym
import torch
from torch.optim import Adam

HERE=Path(__file__).resolve();REPO=HERE.parents[4]
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_yaw_conditioned_omnidirectional_walk"
PARENT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
EXPECTED="bd60f339c2ac8b193cc5fd4063fea8e1acf520740eb94979ffee6097c63d0244";LR=1.5e-5;SAVES={1,10,20,40,60,80,100,120,140,160,180,200}
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parent.parent/"src"))
import isaaclab_tasks,g1_omnidirectional.tasks_w1b
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
from rsl_rl.runners import OnPolicyRunner
p=argparse.ArgumentParser();p.add_argument("--mode",choices=("preflight","train"),required=True);add_launcher_args(p);a,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0],*hydra]
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvw(n,rows):
 f=[]
 for r in rows:
  for k,v in r.items():
   if k not in f and not isinstance(v,(dict,list)):f.append(k)
 with (OUT/n).open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,f,extrasaction="ignore");w.writeheader();w.writerows(rows)
def strict(r):
 x=torch.load(PARENT,map_location=r.device,weights_only=False);r.alg.actor.load_state_dict(x["actor_state_dict"],strict=True);r.alg.critic.load_state_dict(x["critic_state_dict"],strict=True)
 mean=[q for n,q in r.alg.actor.named_parameters() if q.requires_grad and not n.startswith("distribution.")]
 r.alg.optimizer=Adam([{"params":mean,"lr":LR,"name":"actor_mean"},{"params":list(r.alg.critic.parameters()),"lr":LR,"name":"critic"}],lr=LR);r.alg.optimizer.load_state_dict(x["optimizer_state_dict"]);r.alg.learning_rate=LR
 for g in r.alg.optimizer.param_groups:g["lr"]=LR
 r.alg.actor.distribution.log_std_walk.requires_grad_(False);r.alg.actor.distribution.log_std_run.requires_grad_(False)
 steps=sorted({int(v["step"]) for v in r.alg.optimizer.state.values() if "step" in v})
 ae=all(torch.equal(r.alg.actor.state_dict()[k].cpu(),v.cpu()) for k,v in x["actor_state_dict"].items());ce=all(torch.equal(r.alg.critic.state_dict()[k].cpu(),v.cpu()) for k,v in x["critic_state_dict"].items())
 return x,ae,ce,steps
def conditional_gradients(r):
 st=r.alg.storage;obs=st.observations.flatten(0,1);act=st.actions.flatten(0,1);old=st.actions_log_prob.flatten(0,1).squeeze(-1);adv=st.advantages.flatten(0,1).squeeze(-1);policy=obs["policy"];yaw=policy[:,11];pars=[q for n,q in r.alg.actor.named_parameters() if q.requires_grad and not n.startswith("distribution.")]
 r.alg.actor(obs,stochastic_output=True);logp=r.alg.actor.get_output_log_prob(act);ratio=torch.exp(logp-old)
 def vec(mask):
  loss=-(adv[mask]*ratio[mask]).mean();g=torch.autograd.grad(loss,pars,retain_graph=True,allow_unused=True);return torch.cat([x.reshape(-1) for x in g if x is not None]),float(loss)
 gy,ly=vec(yaw.abs()>.02);gt,lt=vec(yaw.abs()<=.02);cos=float(torch.nn.functional.cosine_similarity(gy,gt,dim=0))
 return {"yaw_reward_gradient":float(gy.norm()),"translation_reward_gradient":float(gt.norm()),"yaw_translation_gradient_cosine":cos,"yaw_conditioned_surrogate":ly,"zero_yaw_surrogate":lt}
def probe(r,env):
 c=env.command;c.external_override_enabled=True;obs,_=env.reset();obs=obs.to(r.device);conds=[(.3,i*22.5,0) for i in range(16)]+[(.6,0,0),(1.2,0,0),(0,0,-.3),(0,0,.3),(.3,0,-.3),(.3,0,.3)]
 n=env.num_envs;idx=torch.arange(n,device=r.device)%len(conds);vx=torch.zeros(n,device=r.device);vy=vx.clone();yc=vx.clone()
 for i,(s,d,y) in enumerate(conds):m=idx==i;vx[m]=s*math.cos(math.radians(d));vy[m]=s*math.sin(math.radians(d));yc[m]=y
 robot=env.unwrapped.scene["robot"];sensor=env.unwrapped.scene.sensors["contact_forces"];feet=[i for i,nm in enumerate(sensor.body_names) if "ankle_roll" in nm];rf=[robot.body_names.index(sensor.body_names[i]) for i in feet];steps=round(8/env.unwrapped.step_dt)
 vec=torch.zeros(n,device=r.device);dire=vec.clone();ye=vec.clone();ay=vec.clone();flight=vec.clone();fall=torch.zeros(n,dtype=torch.bool,device=r.device);slip=fall.clone();impact=fall.clone();streak=torch.zeros(n,dtype=torch.long,device=r.device)
 for st in range(steps):
  c.external_override[:,0]=vx;c.external_override[:,1]=vy;c.external_override[:,2]=yc
  if st==0:c._update_command();obs=env.get_observations().to(r.device)
  with torch.inference_mode():action=r.alg.actor(obs)
  obs,_,done,extra=env.step(action);obs=obs.to(r.device);av=robot.data.root_lin_vel_b[:,:2];az=robot.data.root_ang_vel_b[:,2];vec+=torch.linalg.vector_norm(av-torch.stack((vx,vy),1),dim=-1);dire+=torch.atan2(torch.sin(torch.atan2(av[:,1],av[:,0])-torch.atan2(vy,vx)),torch.cos(torch.atan2(av[:,1],av[:,0])-torch.atan2(vy,vx))).abs()*180/math.pi;ye+=(az-yc).abs();ay+=az;force=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);contact=force>5;flight+=(contact.sum(-1)==0).float();timeout=extra.get("time_outs",torch.zeros_like(done)).bool();fall|=done.bool()&~timeout;fs=torch.linalg.vector_norm(robot.data.body_lin_vel_w[:,rf,:2],dim=-1);ss=((fs>.55)&contact).any(-1);streak=torch.where(ss,streak+1,torch.zeros_like(streak));slip|=streak>=5;impact|=force.amax(-1)>3500
 vec/=steps;dire/=steps;ye/=steps;ay/=steps;rates=[]
 for i,(s,d,y) in enumerate(conds):
  m=idx==i;safe=~fall[m]&~slip[m]&~impact[m]&(flight[m]/steps<.1)
  if s==0:ok=safe&(ay[m]*y>0)
  elif y==0:ok=safe&(vec[m]<=.2)&(dire[m]<=20)&(ay[m].abs()<=.2)
  else:ok=safe&(vec[m]<=.25)&(dire[m]<=25)&(ye[m]<=.2)&(ay[m]*y>0)
  rates.append(float(ok.float().mean()))
 out={"zero_yaw_pass_directions":sum(x>=.9 for x in rates[:16]),"forward_0p6_success":rates[16],"forward_1p2_success":rates[17],"pure_left_sign_success":rates[19],"pure_right_sign_success":rates[18],"moving_left_success":rates[21],"moving_right_success":rates[20],"quick_fall_rate":float(fall.float().mean()),"quick_slip_rate":float(slip.float().mean()),"quick_impact_rate":float(impact.float().mean())}
 c.external_override_enabled=False;obs,_=env.reset();return obs.to(r.device),out
def rewards(env):
 rm=env.unwrapped.reward_manager;step=getattr(rm,"_step_reward",None);return {f"reward_{n}":float(step[:,i].mean()) for i,n in enumerate(rm.active_terms)} if step is not None else {}
def save(r,path,it,row):
 x=r.alg.save();x["iter"]=it;x["infos"]={"experiment":"exp_013","phase":"W1B","training_iteration":it,"curriculum_phase":row.get("curriculum_phase","INITIAL"),"learning_rate":LR,"parent":"W1A2 iteration 80","rollout_kl":row.get("exact_rollout_kl"),"clip_fraction":row.get("clip_fraction"),"yaw_tracking":row.get("reward_track_ang_vel_z_exp"),"zero_yaw_retention":row.get("zero_yaw_pass_directions")};torch.save(x,path)
def main():
 if sha(PARENT)!=EXPECTED:raise RuntimeError("EXP013_W1B_STRICT_RESUME_FAIL")
 cfg,ac=resolve_task_config("Isaac-Exp013-G1-W1B-YawWalk-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=1024;cfg.seed=ac.seed=20274021;ac.max_iterations=200
 if a.device:cfg.sim.device=ac.device=a.device
 with launch_simulation(cfg,a):
  saved=sys.argv[:];sys.argv=["train_w1a.py","--mode","preflight","--headless"];import train_w1a as w1;sys.argv=saved
  base=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-W1B-YawWalk-v0",cfg=cfg),clip_actions=ac.clip_actions);env=w1.W1AVecEnv(base)
  import importlib.metadata;ac=handle_deprecated_rsl_rl_cfg(ac,importlib.metadata.version("rsl-rl-lib"));r=OnPolicyRunner(env,ac.to_dict(),log_dir=None,device=ac.device);src,ae,ce,steps=strict(r)
  if not(ae and ce and steps==[4000]):raise RuntimeError("EXP013_W1B_STRICT_RESUME_FAIL")
  walk0=r.alg.actor.distribution.log_std_walk.detach().clone();run0=r.alg.actor.distribution.log_std_run.detach().clone();obs,_=env.reset();obs=obs.to(r.device);env.command.set_training_iteration(1);env.command._resample_command(torch.arange(env.num_envs,device=r.device));obs=env.get_observations().to(r.device);obs,safety=w1.rollout(r,env,obs);cg=conditional_gradients(r);loss,m=w1.update_once(r)
  first={**m,**safety,**cg,"actor_gradient":w1.grad_norm(r.alg.actor),"critic_gradient":w1.grad_norm(r.alg.critic),"value_loss":float(loss.get("value",0)),"nan_inf":0 if all(torch.isfinite(q).all() for q in r.alg.actor.parameters()) else 1,"optimizer_lr":[g["lr"] for g in r.alg.optimizer.param_groups],"runtime_lr":r.alg.learning_rate}
  ok=first["exact_rollout_kl"]<=.2 and first["all_step_maximum_kl"]<=.2 and first["clip_fraction"]<=.5 and first["mean_action_shift"]<=2 and first["critic_gradient"]<=1e6 and first["value_loss"]<=1e8 and first["nan_inf"]==0 and all(abs(x-LR)<1e-12 for x in first["optimizer_lr"]);first["status"]="PASS" if ok else "EXP013_W1B_TRAINING_UNSTABLE";first["preferred_kl"]=first["exact_rollout_kl"]<=.05;first["preferred_clip"]=first["clip_fraction"]<=.3
  if a.mode=="preflight":dump("first_update_stability.json",first);g=json.loads((OUT/"gate.json").read_text());g.update({"first_update":"PASS" if ok else "FAIL","continue_training":ok});dump("gate.json",g);env.close();return
  if json.loads((OUT/"first_update_stability.json").read_text())["status"]!="PASS":raise RuntimeError("PREFLIGHT_NOT_PASS")
  src,ae,ce,steps=strict(r);ck=OUT/"checkpoints";ck.mkdir(exist_ok=True);save(r,ck/"model_initial.pt",0,{})
  obs,_=env.reset();obs=obs.to(r.device);curves=[];early=[];stopped=None
  for it in range(1,201):
   env.command.set_training_iteration(it);env.command._resample_command(torch.arange(env.num_envs,device=r.device));obs=env.get_observations().to(r.device);obs,safety=w1.rollout(r,env,obs);rt=rewards(env);loss,m=w1.update_once(r);finite=all(torch.isfinite(q).all() for q in r.alg.actor.parameters())
   row={"iteration":it,"interactions":it*1024*24,"curriculum_phase":env.command.phase,**safety,**{k:v for k,v in m.items() if k!="optimizer_step_trace"},**rt,"actor_gradient":w1.grad_norm(r.alg.actor),"critic_gradient":w1.grad_norm(r.alg.critic),"value_loss":float(loss.get("value",0)),"learning_rate":r.alg.learning_rate,"nan_inf":0 if finite else 1,"std_frozen":torch.equal(walk0,r.alg.actor.distribution.log_std_walk) and torch.equal(run0,r.alg.actor.distribution.log_std_run)}
   if it<=10:
    obs,q=probe(r,env);row.update(q);fail=(not finite or row["exact_rollout_kl"]>.5 or safety["fall_rate"]>.1 or safety["dangerous_slip_rate"]>.35 or safety["impact_failure_rate"]>.1 or q["zero_yaw_pass_directions"]<12 or q["forward_0p6_success"]<.85 or q["forward_1p2_success"]<.85 or min(q["pure_left_sign_success"],q["pure_right_sign_success"])<.8);row["guard_pass"]=not fail;early.append(row.copy());stopped="EXP013_W1B_TRAINING_UNSTABLE" if fail else stopped
   curves.append(row)
   if it in SAVES or stopped:save(r,ck/f"model_{it}.pt",it,row)
   print(f"[W1B] iter={it} phase={env.command.phase} kl={row['exact_rollout_kl']:.5f} fall={safety['fall_rate']:.3f} yawR={rt.get('reward_track_ang_vel_z_exp',0):.3f}",flush=True)
   if stopped:break
  csvw("training_curves.csv",curves);dump("early_guard.json",{"status":"PASS" if not stopped and len(early)==10 else stopped,"rows":early});dump("training_run_summary.json",{"status":stopped or "COMPLETE","iterations":len(curves),"interactions":len(curves)*1024*24,"maximum_runs":1})
  g=json.loads((OUT/"gate.json").read_text());g.update({"training":"COMPLETE" if not stopped else "STOPPED","continue_evaluation":not stopped and len(curves)==200});dump("gate.json",g);env.close()
  if stopped:raise SystemExit(3)
if __name__=="__main__":main()
