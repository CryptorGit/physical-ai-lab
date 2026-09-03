"""Strict-resume W1A2 preflight and the single authorized PPO continuation."""
from __future__ import annotations
import argparse,csv,hashlib,io,json,math,sys
from pathlib import Path
import gymnasium as gym
import torch
from torch.optim import Adam

HERE=Path(__file__).resolve(); EXP=HERE.parent.parent; REPO=HERE.parents[4]
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion"
PARENT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a_all_direction_translation_walk/checkpoints/model_120.pt"
EXPECTED="b128f6b164d151b411eeaf2caf22edc1ea2a69e68fca9534e7d6a965ae4dbba9"; LR=1.5e-5
SAVES={1,10,20,40,60,80,100,120,140,160}
sys.path.insert(0,str(HERE.parent))
import isaaclab_tasks
import g1_omnidirectional.tasks_w1a2
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
from rsl_rl.runners import OnPolicyRunner
parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=("preflight","train"),required=True);add_launcher_args(parser)
args,hydra=setup_preset_cli(parser);sys.argv=[sys.argv[0],*hydra]
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvwrite(n,rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with (OUT/n).open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def strict_load(runner):
 x=torch.load(PARENT,map_location=runner.device,weights_only=False)
 runner.alg.actor.load_state_dict(x["actor_state_dict"],strict=True);runner.alg.critic.load_state_dict(x["critic_state_dict"],strict=True)
 runner.alg.optimizer=Adam([{"params":[p for n,p in runner.alg.actor.named_parameters() if p.requires_grad and not n.startswith("distribution.")],"lr":LR,"name":"actor_mean"},{"params":list(runner.alg.critic.parameters()),"lr":LR,"name":"critic"}],lr=LR)
 runner.alg.optimizer.load_state_dict(x["optimizer_state_dict"]);runner.alg.learning_rate=LR
 for g in runner.alg.optimizer.param_groups:g["lr"]=LR
 runner.alg.actor.distribution.log_std_walk.requires_grad_(False);runner.alg.actor.distribution.log_std_run.requires_grad_(False)
 actor=all(torch.equal(runner.alg.actor.state_dict()[k].cpu(),v.cpu()) for k,v in x["actor_state_dict"].items())
 critic=all(torch.equal(runner.alg.critic.state_dict()[k].cpu(),v.cpu()) for k,v in x["critic_state_dict"].items())
 steps=sorted({int(v["step"]) for v in runner.alg.optimizer.state.values() if "step" in v})
 return x,actor,critic,steps
def probe(runner,env):
 command=env.command;command.external_override_enabled=True;obs,_=env.reset();obs=obs.to(runner.device)
 conditions=[(.3,d) for d in range(0,360,22)][:0]
 conditions=[(.3,i*22.5) for i in range(16)]+[(.6,0),(1.2,0)]
 n=env.num_envs; idx=torch.arange(n,device=runner.device)%len(conditions)
 vx=torch.zeros(n,device=runner.device);vy=torch.zeros_like(vx)
 for i,(s,d) in enumerate(conditions):
  mask=idx==i;vx[mask]=s*math.cos(math.radians(d));vy[mask]=s*math.sin(math.radians(d))
 robot=env.unwrapped.scene["robot"];sensor=env.unwrapped.scene.sensors["contact_forces"];feet=[i for i,nm in enumerate(sensor.body_names) if "ankle_roll" in nm]
 rfeet=[robot.body_names.index(sensor.body_names[i]) for i in feet];steps=round(8/env.unwrapped.step_dt)
 err=torch.zeros(n,device=runner.device);falls=torch.zeros(n,dtype=torch.bool,device=runner.device);danger=torch.zeros_like(falls);impact=torch.zeros_like(falls);flight=torch.zeros(n,device=runner.device);streak=torch.zeros(n,dtype=torch.long,device=runner.device)
 for step in range(steps):
  command.external_override[:,0]=vx;command.external_override[:,1]=vy;command.external_override[:,2]=0
  if step==0:command._update_command();obs=env.get_observations().to(runner.device)
  with torch.inference_mode():action=runner.alg.actor(obs)
  obs,_,done,extras=env.step(action);obs=obs.to(runner.device);actual=robot.data.root_lin_vel_b[:,:2]
  err+=torch.linalg.vector_norm(actual-torch.stack((vx,vy),1),dim=-1)
  forces=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);contacts=forces>5;flight+=(contacts.sum(-1)==0).float()
  timeout=extras.get("time_outs",torch.zeros_like(done)).bool();falls|=done.bool()&~timeout
  fs=torch.linalg.vector_norm(robot.data.body_lin_vel_w[:,rfeet,:2],dim=-1);slip=((fs>.55)&contacts).any(-1);streak=torch.where(slip,streak+1,torch.zeros_like(streak));danger|=streak>=5;impact|=forces.amax(-1)>3500
 err/=steps;flight/=steps;result={}
 passes=0;means=[]
 for i,(s,d) in enumerate(conditions):
  mask=idx==i;success=(err[mask]<=.20)&(flight[mask]<.10)&~falls[mask]&~danger[mask]&~impact[mask]
  rate=float(success.float().mean());mae=float(err[mask].mean());result[f"S{s:.1f}_D{d:05.1f}_success"]=rate;result[f"S{s:.1f}_D{d:05.1f}_mae"]=mae
  if s==.3:passes+=rate>=.90;means.append((d,mae))
 mirror=[]
 for d in (22.5,45,67.5,90,112.5,135,157.5):
  mirror.append(abs(dict(means)[d]-dict(means)[360-d]))
 result.update({"pass_0p3_directions":passes,"forward_0p6_success":result["S0.6_D000.0_success"],"forward_1p2_success":result["S1.2_D000.0_success"],"mirror_mae_difference":sum(mirror)/len(mirror)})
 command.external_override_enabled=False;obs,_=env.reset();return obs.to(runner.device),result
def reward_snapshot(env):
 rm=env.unwrapped.reward_manager;values={}
 step=getattr(rm,"_step_reward",None)
 if step is not None:
  for i,name in enumerate(rm.active_terms):values[f"reward_{name}"]=float(step[:,i].mean())
 return values
def save(runner,path,it,phase,curve):
 p=runner.alg.save();p["iter"]=it;p["infos"]={"experiment":"exp_013","phase":"W1A2","training_iteration":it,"curriculum_phase":phase,"learning_rate":LR,"parent_w1a_iteration":120,"rollout_kl":curve.get("exact_rollout_kl"),"clip_fraction":curve.get("clip_fraction")};torch.save(p,path)
def main():
 if sha(PARENT)!=EXPECTED:raise RuntimeError("EXP013_W1A2_STRICT_RESUME_FAIL")
 cfg,acfg=resolve_task_config("Isaac-Exp013-G1-W1A2-SpeedEnvelope-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=1024;cfg.seed=20272021;acfg.seed=20272021;acfg.max_iterations=160
 if args.device:cfg.sim.device=acfg.device=args.device
 with launch_simulation(cfg,args):
  global w1
  saved_argv=sys.argv[:];sys.argv=["train_w1a.py","--mode","preflight","--headless"]
  import train_w1a as w1
  sys.argv=saved_argv
  raw=gym.make("Isaac-Exp013-G1-W1A2-SpeedEnvelope-v0",cfg=cfg);base=RslRlVecEnvWrapper(raw,clip_actions=acfg.clip_actions);env=w1.W1AVecEnv(base)
  import importlib.metadata;acfg=handle_deprecated_rsl_rl_cfg(acfg,importlib.metadata.version("rsl-rl-lib"))
  runner=OnPolicyRunner(env,acfg.to_dict(),log_dir=None,device=acfg.device);source,ae,ce,adam_steps=strict_load(runner)
  if not(ae and ce and adam_steps==[2400]):raise RuntimeError("EXP013_W1A2_STRICT_RESUME_FAIL")
  walk0=runner.alg.actor.distribution.log_std_walk.detach().clone();run0=runner.alg.actor.distribution.log_std_run.detach().clone()
  obs,_=env.reset();obs=obs.to(runner.device);env.command.set_training_iteration(1);env.command._resample_command(torch.arange(env.num_envs,device=runner.device));obs=env.get_observations().to(runner.device)
  obs,safety=w1.rollout(runner,env,obs);losses,metrics=w1.update_once(runner)
  finite=all(torch.isfinite(p).all() for p in runner.alg.actor.parameters());first={**metrics,**safety,"actor_gradient":w1.grad_norm(runner.alg.actor),"critic_gradient":w1.grad_norm(runner.alg.critic),"value_loss":float(losses.get("value",0)),"surrogate_improvement":metrics["surrogate_improvement"],"nan_inf":0 if finite else 1,"optimizer_lr":[g["lr"] for g in runner.alg.optimizer.param_groups],"runtime_lr":runner.alg.learning_rate}
  passed=first["exact_rollout_kl"]<=.2 and first["all_step_maximum_kl"]<=.2 and first["clip_fraction"]<=.5 and first["mean_action_shift"]<=2 and first["critic_gradient"]<=1e6 and first["value_loss"]<=1e8 and first["nan_inf"]==0 and all(abs(x-LR)<1e-12 for x in first["optimizer_lr"])
  first["status"]="PASS" if passed else "EXP013_W1A2_TRAINING_UNSTABLE";first["preferred_kl"]=first["exact_rollout_kl"]<=.05;first["preferred_clip"]=first["clip_fraction"]<=.30
  if args.mode=="preflight":
   dump("first_update_stability.json",first);g=json.loads((OUT/"gate.json").read_text());g.update({"first_update":"PASS" if passed else "FAIL","continue_training":passed});dump("gate.json",g);print(json.dumps(first));env.close();raise SystemExit(0 if passed else 2)
  if json.loads((OUT/"first_update_stability.json").read_text())["status"]!="PASS":raise RuntimeError("PREFLIGHT_NOT_PASS")
  source,ae,ce,adam_steps=strict_load(runner);check=OUT/"checkpoints";check.mkdir(exist_ok=True);save(runner,check/"model_initial.pt",0,"INITIAL",{})
  obs,_=env.reset();obs=obs.to(runner.device);curves=[];early=[];stopped=None
  for it in range(1,161):
   env.command.set_training_iteration(it);env.command._resample_command(torch.arange(env.num_envs,device=runner.device));obs=env.get_observations().to(runner.device)
   obs,safety=w1.rollout(runner,env,obs);reward_terms=reward_snapshot(env);losses,metrics=w1.update_once(runner)
   finite=all(torch.isfinite(p).all() for p in runner.alg.actor.parameters());stdok=torch.equal(walk0,runner.alg.actor.distribution.log_std_walk) and torch.equal(run0,runner.alg.actor.distribution.log_std_run)
   row={"iteration":it,"interactions":it*1024*24,"curriculum_phase":env.command.phase,**safety,**{k:v for k,v in metrics.items() if k!="optimizer_step_trace"},**reward_terms,"actor_gradient":w1.grad_norm(runner.alg.actor),"critic_gradient":w1.grad_norm(runner.alg.critic),"value_loss":float(losses.get("value",0)),"learning_rate":runner.alg.learning_rate,"nan_inf":0 if finite else 1,"std_frozen":stdok}
   if it<=10:
    obs,p=probe(runner,env);row.update(p);fail=(not finite or row["exact_rollout_kl"]>.5 or safety["fall_rate"]>.10 or safety["dangerous_slip_rate"]>.30 or safety["impact_failure_rate"]>.10 or p["pass_0p3_directions"]<14 or p["forward_0p6_success"]<.90 or p["forward_1p2_success"]<.90 or p["mirror_mae_difference"]>.20);row["guard_pass"]=not fail;early.append(row.copy())
    if fail:stopped="EXP013_W1A2_TRAINING_UNSTABLE"
   curves.append(row)
   if it in SAVES or stopped:save(runner,check/f"model_{it}.pt",it,env.command.phase,row)
   print(f"[W1A2] iter={it} phase={env.command.phase} reward={safety['mean_reward']:.3f} fall={safety['fall_rate']:.3f} slip={safety['dangerous_slip_rate']:.3f} kl={row['exact_rollout_kl']:.5f} clip={row['clip_fraction']:.3f}",flush=True)
   if stopped:break
  csvwrite("training_curves.csv",curves);dump("early_guard.json",{"status":"PASS" if not stopped and len(early)==10 else stopped,"rows":early});dump("training_run_summary.json",{"status":stopped or "COMPLETE","iterations":len(curves),"interactions":len(curves)*1024*24,"maximum_runs":1,"strict_resume_adam_step":2400})
  g=json.loads((OUT/"gate.json").read_text());g.update({"training":"COMPLETE" if not stopped else "STOPPED","continue_evaluation":not stopped and len(curves)==160});dump("gate.json",g);env.close()
  if stopped:raise SystemExit(3)
if __name__=="__main__":main()
