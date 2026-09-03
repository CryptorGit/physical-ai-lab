"""Retention-beta shadow preflight and the single authorized W1A4 PPO run."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
import gymnasium as gym
import torch
from torch import nn
from torch.optim import Adam
from tensordict import TensorDict

HERE=Path(__file__).resolve();REPO=HERE.parents[4]
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a4_low_speed_retention_consolidation"
PARENT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
EXPECTED="bd60f339c2ac8b193cc5fd4063fea8e1acf520740eb94979ffee6097c63d0244";LR=1.5e-5
SAVES={1,5,10,20,30,40,50,60}
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parent.parent/"src"))
import isaaclab_tasks,g1_omnidirectional.tasks_w1a4
from g1_single_policy.phase_gated_heading import yaw_from_quat_wxyz
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
from rsl_rl.runners import OnPolicyRunner
parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=("preflight","train"),required=True);add_launcher_args(parser)
args,hydra=setup_preset_cli(parser);sys.argv=[sys.argv[0],*hydra]
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def csvwrite(n,rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with (OUT/n).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def grad_norm(params):return math.sqrt(sum(float((p.grad.detach()**2).sum()) for p in params if p.grad is not None))
def strict_load(runner):
 p=torch.load(PARENT,map_location=runner.device,weights_only=False)
 runner.alg.actor.load_state_dict(p["actor_state_dict"],strict=True);runner.alg.critic.load_state_dict(p["critic_state_dict"],strict=True)
 mean=[x for n,x in runner.alg.actor.named_parameters() if x.requires_grad and not n.startswith("distribution.")]
 runner.alg.optimizer=Adam([{"params":mean,"lr":LR,"name":"actor_mean"},{"params":list(runner.alg.critic.parameters()),"lr":LR,"name":"critic"}],lr=LR)
 runner.alg.optimizer.load_state_dict(p["optimizer_state_dict"]);runner.alg.learning_rate=LR
 for g in runner.alg.optimizer.param_groups:g["lr"]=LR
 runner.alg.actor.distribution.log_std_walk.requires_grad_(False);runner.alg.actor.distribution.log_std_run.requires_grad_(False)
 steps=sorted({int(v["step"]) for v in runner.alg.optimizer.state.values() if "step" in v})
 return p,mean,steps
class RetentionHook:
 def __init__(self,runner,anchor,beta,seed):
  self.r=runner;self.a=anchor;self.beta=beta;self.gen=torch.Generator(device=runner.device).manual_seed(seed);self.rows=[]
  self.params=[p for n,p in runner.alg.actor.named_parameters() if p.requires_grad and not n.startswith("distribution.")]
  self.original=runner.alg.optimizer.step;runner.alg.optimizer.step=self.step
 def step(self,*a,**k):
  ppo=grad_norm(self.params);n=min(2048,len(self.a["train_observations"]))
  idx=torch.randint(len(self.a["train_observations"]),(n,),generator=self.gen,device=self.r.device)
  obs=self.a["train_observations"][idx];ref=self.a["train_reference_mean"][idx]
  actor_obs=TensorDict({"policy":obs},batch_size=[n],device=self.r.device)
  current=self.r.alg.actor(actor_obs);std=self.r.alg.actor.distribution.log_std_walk.exp()
  kl=.5*(((ref-current)/std)**2).sum(-1).mean()
  grads=torch.autograd.grad(kl,self.params,retain_graph=False,allow_unused=True)
  retention=math.sqrt(sum(float((g.detach()**2).sum()) for g in grads if g is not None))
  for p,g in zip(self.params,grads):
   if g is not None:p.grad.add_(g,alpha=self.beta)
  combined=grad_norm(self.params);nn.utils.clip_grad_norm_(self.params,self.r.alg.max_grad_norm)
  self.rows.append({"retention_loss":float(kl.detach()),"ppo_gradient_norm":ppo,"retention_gradient_norm":retention,
   "effective_retention_ppo_ratio":self.beta*retention/max(ppo,1e-12),"combined_gradient_norm":combined})
  return self.original(*a,**k)
 def close(self):self.r.alg.optimizer.step=self.original
 def mean(self,key):return sum(x[key] for x in self.rows)/max(len(self.rows),1)
def holdout_kl(runner,anchor):
 with torch.no_grad():
  obs=anchor["holdout_observations"];ref=anchor["holdout_reference_mean"]
  actor_obs=TensorDict({"policy":obs},batch_size=[obs.shape[0]],device=runner.device)
  cur=runner.alg.actor(actor_obs);std=runner.alg.actor.distribution.log_std_walk.exp()
  return float((.5*(((ref-cur)/std)**2).sum(-1)).mean())
def parameter_change(runner,initial):
 return math.sqrt(sum(float(((v.detach()-initial[k])**2).sum()) for k,v in runner.alg.actor.state_dict().items() if k in initial))
def quick_probe(runner,env):
 command=env.command;command.external_override_enabled=True;obs,_=env.reset();obs=obs.to(runner.device)
 conditions=[(.3,i*22.5) for i in range(16)]+[(.6,i*22.5) for i in range(16)]+[(1.2,0)]
 per=20;used=len(conditions)*per;ids=torch.arange(env.num_envs,device=runner.device)//per;active=torch.arange(env.num_envs,device=runner.device)<used
 ids=ids.clamp_max(len(conditions)-1);vx=torch.zeros(env.num_envs,device=runner.device);vy=torch.zeros_like(vx)
 for i,(s,d) in enumerate(conditions):
  mask=(ids==i)&active;vx[mask]=s*math.cos(math.radians(d));vy[mask]=s*math.sin(math.radians(d))
 robot=env.unwrapped.scene["robot"];sensor=env.unwrapped.scene.sensors["contact_forces"];feet=[i for i,n in enumerate(sensor.body_names) if "ankle_roll" in n]
 rfeet=[robot.body_names.index(sensor.body_names[i]) for i in feet];steps=round(8/env.unwrapped.step_dt)
 err=torch.zeros(env.num_envs,device=runner.device);derr=torch.zeros_like(err);yaw=torch.zeros_like(err);flight=torch.zeros_like(err)
 fall=torch.zeros(env.num_envs,dtype=torch.bool,device=runner.device);danger=fall.clone();impact=fall.clone();sat=fall.clone()
 slipst=torch.zeros(env.num_envs,dtype=torch.long,device=runner.device);satst=slipst.clone();initial_yaw=yaw_from_quat_wxyz(robot.data.root_quat_w).clone()
 heading_history=torch.zeros((steps,used),device=runner.device)
 for step in range(steps):
  command.external_override[:,0]=vx;command.external_override[:,1]=vy;command.external_override[:,2]=0
  if step==0:command._update_command();obs=env.get_observations().to(runner.device)
  with torch.inference_mode():predicted_action=runner.alg.actor(obs)
  action=torch.where(active[:,None],predicted_action,torch.zeros_like(predicted_action))
  obs,_,done,extras=env.step(action);obs=obs.to(runner.device)
  actual=robot.data.root_lin_vel_b[:,:2];err+=torch.linalg.vector_norm(actual-torch.stack((vx,vy),1),dim=-1)
  derr+=torch.atan2(torch.sin(torch.atan2(actual[:,1],actual[:,0])-torch.atan2(vy,vx)),torch.cos(torch.atan2(actual[:,1],actual[:,0])-torch.atan2(vy,vx))).abs()*180/math.pi
  yaw+=robot.data.root_ang_vel_b[:,2].abs();forces=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);contacts=forces>5;flight+=(contacts.sum(-1)==0).float()
  timeout=extras.get("time_outs",torch.zeros_like(done)).bool();fall|=done.bool()&~timeout&active
  fs=torch.linalg.vector_norm(robot.data.body_lin_vel_w[:,rfeet,:2],dim=-1);sl=((fs>.55)&contacts).any(-1);slipst=torch.where(sl,slipst+1,torch.zeros_like(slipst));danger|=(slipst>=5)&active;impact|=(forces.amax(-1)>3500)&active
  limits=robot.data.joint_vel_limits
  if limits.ndim==3:limits=limits[...,1].abs()
  saturated=robot.data.joint_vel.abs().div(limits.clamp_min(1e-6)).amax(-1)>.95;satst=torch.where(saturated,satst+1,torch.zeros_like(satst));sat|=(satst>=5)&active
  heading=torch.atan2(torch.sin(yaw_from_quat_wxyz(robot.data.root_quat_w)-initial_yaw),torch.cos(yaw_from_quat_wxyz(robot.data.root_quat_w)-initial_yaw)).abs()
  heading_history[step]=heading[:used]
 err/=steps;derr/=steps;yaw/=steps;flight/=steps;rates=[];maes=[];direrrs=[]
 for i,c in enumerate(conditions):
  mask=(ids==i)&active;env_ids=torch.where(mask)[0];hp=torch.quantile(heading_history[:,env_ids],.95,dim=0)
  success=(err[mask]<=.2)&(derr[mask]<=20)&(yaw[mask]<=.2)&(flight[mask]<.1)&(hp<=.25)&~fall[mask]&~danger[mask]&~impact[mask]&~sat[mask]
  rates.append(float(success.float().mean()));maes.append(float(err[mask].mean()));direrrs.append(float(derr[mask].mean()))
 command.external_override_enabled=False;obs,_=env.reset()
 return obs.to(runner.device),{"pass_0p3":sum(x>=.9 for x in rates[:16]),"pass_0p6":sum(x>=.9 for x in rates[16:32]),
  "success_225_0p3":rates[10],"success_247p5_0p3":rates[11],"forward_0p6_success":rates[16],"forward_1p2_success":rates[32],
  "rear_left_direction_error":(direrrs[10]+direrrs[11])/2,"probe_fall_rate":float(fall[active].float().mean()),
  "probe_dangerous_slip_rate":float(danger[active].float().mean()),"probe_impact_rate":float(impact[active].float().mean())}
def save(runner,path,it,beta,row):
 p=runner.alg.save();p["iter"]=it;p["infos"]={"experiment":"exp_013","phase":"W1A4","training_iteration":it,"beta":beta,"learning_rate":LR,
 "low_speed_holdout_kl":row.get("low_speed_holdout_kl"),"rollout_kl":row.get("exact_rollout_kl"),"clip_fraction":row.get("clip_fraction"),"parent":"W1A2 iteration 80"};torch.save(p,path)
def main():
 if sha(PARENT)!=EXPECTED:raise RuntimeError("EXP013_W1A4_STRICT_RESUME_FAIL")
 cache=torch.load(OUT/"low_speed_anchor_cache.pt",map_location="cpu",weights_only=False)
 cfg,acfg=resolve_task_config("Isaac-Exp013-G1-W1A4-Retention-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=1024;cfg.seed=acfg.seed=20273021;acfg.max_iterations=60
 if args.device:cfg.sim.device=acfg.device=args.device
 with launch_simulation(cfg,args):
  saved=sys.argv[:];sys.argv=["train_w1a.py","--mode","preflight","--headless"];import train_w1a as w1;sys.argv=saved
  raw=gym.make("Isaac-Exp013-G1-W1A4-Retention-v0",cfg=cfg);base=RslRlVecEnvWrapper(raw,clip_actions=acfg.clip_actions);env=w1.W1AVecEnv(base)
  import importlib.metadata;acfg=handle_deprecated_rsl_rl_cfg(acfg,importlib.metadata.version("rsl-rl-lib"))
  runner=OnPolicyRunner(env,acfg.to_dict(),log_dir=None,device=acfg.device)
  anchor={k:(v.to(runner.device) if torch.is_tensor(v) else v) for k,v in cache.items()}
  # The anchor archive preserves the original 123-D environment observation.
  # Append the fixed WALK gait command to honor the unchanged 124-D actor input.
  for split in ("train","holdout"):
   key=f"{split}_observations"
   if anchor[key].shape[-1]==123:
    anchor[key]=torch.cat((anchor[key],torch.zeros((anchor[key].shape[0],1),device=runner.device)),dim=-1)
  if args.mode=="preflight":
   rows=[]
   for bi,beta in enumerate((0.,.01,.03,.10)):
    torch.manual_seed(20273021);env.seed(20273021);source,params,steps=strict_load(runner)
    if steps!=[4000]:raise RuntimeError("EXP013_W1A4_STRICT_RESUME_FAIL")
    initial={k:v.detach().clone() for k,v in runner.alg.actor.state_dict().items()};obs,_=env.reset();obs=obs.to(runner.device)
    for update in range(1,6):
     env.command.set_training_iteration(update);env.command._resample_command(torch.arange(env.num_envs,device=runner.device));obs=env.get_observations().to(runner.device)
     obs,safety=w1.rollout(runner,env,obs);hook=RetentionHook(runner,anchor,beta,20273021+update)
     losses,metrics=w1.update_once(runner);hook.close();obs,probe=quick_probe(runner,env)
     row={"branch":f"B{int(beta*100):03d}","beta":beta,"update":update,**metrics,**safety,**probe,
      "low_speed_holdout_kl":holdout_kl(runner,anchor),"ppo_gradient_norm":hook.mean("ppo_gradient_norm"),
      "retention_gradient_norm":hook.mean("retention_gradient_norm"),"effective_retention_ppo_ratio":hook.mean("effective_retention_ppo_ratio"),
      "combined_gradient_norm":hook.mean("combined_gradient_norm"),"actor_parameter_change":parameter_change(runner,initial),
      "actor_gradient":grad_norm(runner.alg.actor.parameters()),"critic_gradient":grad_norm(runner.alg.critic.parameters()),
      "value_loss":float(losses["value"]),"critic_improvement":-float(losses["value"]),
      "optimizer_lr":runner.alg.optimizer.param_groups[0]["lr"],"runtime_lr":runner.alg.learning_rate,
      "lr_contract_match":all(abs(g["lr"]-LR)<1e-12 for g in runner.alg.optimizer.param_groups) and abs(runner.alg.learning_rate-LR)<1e-12,
      "nan_inf":0 if all(torch.isfinite(p).all() for p in runner.alg.actor.parameters()) else 1}
     rows.append(row)
   csvwrite("retention_beta_preflight.csv",rows);final=[r for r in rows if r["update"]==5];eligible=[r for r in final if r["low_speed_holdout_kl"]<=.02 and r["pass_0p3"]==16 and r["success_225_0p3"]>=.9 and r["success_247p5_0p3"]>=.9 and r["actor_parameter_change"]>0 and abs(r["surrogate_improvement"])>0 and r["effective_retention_ppo_ratio"]<=.5 and r["nan_inf"]==0]
   if not eligible:
    dump("selected_retention_beta.json",{"status":"EXP013_W1A4_RETENTION_COEFFICIENT_NOT_FOUND","selected_beta":None});g=json.loads((OUT/"gate.json").read_text());g.update({"beta_preflight":"FAIL","continue_training":False});dump("gate.json",g);env.close();raise SystemExit(4)
   selected=min(eligible,key=lambda r:r["beta"]);dump("selected_retention_beta.json",{"status":"PASS","selected_beta":selected["beta"],"selection_rule":"minimum passing beta","final_update":selected})
   first=next(r for r in rows if r["beta"]==selected["beta"] and r["update"]==1);first["status"]="PASS" if first["exact_rollout_kl"]<=.2 and first["all_step_maximum_kl"]<=.2 and first["clip_fraction"]<=.5 and first["mean_action_shift"]<=2 and first["critic_gradient"]<=1e6 and first["value_loss"]<=1e8 and first["nan_inf"]==0 and first["lr_contract_match"] else "FAIL";dump("first_update_stability.json",first)
   g=json.loads((OUT/"gate.json").read_text());g.update({"beta_preflight":"PASS","one_update":first["status"],"continue_training":first["status"]=="PASS"});dump("gate.json",g);env.close();return
  selected=json.loads((OUT/"selected_retention_beta.json").read_text());beta=selected["selected_beta"]
  if beta is None or json.loads((OUT/"first_update_stability.json").read_text())["status"]!="PASS":raise RuntimeError("PREFLIGHT_NOT_PASS")
  torch.manual_seed(20273021);env.seed(20273021);source,params,steps=strict_load(runner);walk0=runner.alg.actor.distribution.log_std_walk.detach().clone();run0=runner.alg.actor.distribution.log_std_run.detach().clone()
  check=OUT/"checkpoints";check.mkdir(exist_ok=True);save(runner,check/"model_initial.pt",0,beta,{})
  obs,_=env.reset();obs=obs.to(runner.device);curves=[];early=[];stopped=None
  for it in range(1,61):
   env.command.set_training_iteration(it);env.command._resample_command(torch.arange(env.num_envs,device=runner.device));obs=env.get_observations().to(runner.device)
   obs,safety=w1.rollout(runner,env,obs);hook=RetentionHook(runner,anchor,beta,20273021+it);losses,metrics=w1.update_once(runner);hook.close()
   row={"iteration":it,"interactions":it*1024*24,"beta":beta,**metrics,**safety,"low_speed_holdout_kl":holdout_kl(runner,anchor),
    "ppo_gradient_norm":hook.mean("ppo_gradient_norm"),"retention_gradient_norm":hook.mean("retention_gradient_norm"),"effective_retention_ppo_ratio":hook.mean("effective_retention_ppo_ratio"),
    "combined_gradient_norm":hook.mean("combined_gradient_norm"),"value_loss":float(losses["value"]),"learning_rate":runner.alg.learning_rate,
    "nan_inf":0 if all(torch.isfinite(p).all() for p in runner.alg.actor.parameters()) else 1,
    "std_frozen":torch.equal(walk0,runner.alg.actor.distribution.log_std_walk) and torch.equal(run0,runner.alg.actor.distribution.log_std_run)}
   if it<=10:
    obs,probe=quick_probe(runner,env);row.update(probe);fail=(row["nan_inf"]>0 or row["exact_rollout_kl"]>.5 or row["low_speed_holdout_kl"]>.03 or safety["fall_rate"]>.1 or safety["dangerous_slip_rate"]>.3 or safety["impact_failure_rate"]>.1 or probe["forward_0p6_success"]<.9 or probe["forward_1p2_success"]<.9 or probe["success_225_0p3"]<.8 or probe["success_247p5_0p3"]<.8);row["guard_pass"]=not fail;early.append(row.copy())
    if fail:stopped="EXP013_W1A4_TRAINING_UNSTABLE"
   curves.append(row)
   if it in SAVES or stopped:save(runner,check/f"model_{it}.pt",it,beta,row)
   print(f"[W1A4] iter={it} beta={beta} kl={row['exact_rollout_kl']:.5f} hold={row['low_speed_holdout_kl']:.5f} fall={safety['fall_rate']:.3f}",flush=True)
   if stopped:break
  csvwrite("training_curves.csv",curves);dump("early_guard.json",{"status":"PASS" if not stopped else stopped,"rows":early});dump("training_run_summary.json",{"status":stopped or "COMPLETE","iterations":len(curves),"interactions":len(curves)*1024*24,"maximum_persistent_runs":1})
  g=json.loads((OUT/"gate.json").read_text());g.update({"training":"COMPLETE" if not stopped else "STOPPED","continue_evaluation":not stopped and len(curves)==60});dump("gate.json",g);env.close()
  if stopped:raise SystemExit(5)
if __name__=="__main__":main()
