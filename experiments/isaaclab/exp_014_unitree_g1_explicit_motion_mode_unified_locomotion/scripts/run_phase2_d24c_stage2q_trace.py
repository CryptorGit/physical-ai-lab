"""Exact 10 s exp012 STAND_TO_WALK replay with full saturation trace."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
HERE=Path(__file__).resolve();REPO=HERE.parents[4];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d24c_shold_source_and_safety_parity/raw"
sys.path.insert(0,str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"));sys.path.insert(0,str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
import isaaclab_tasks,g1_single_policy.tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
import importlib.util
def mod(n,p):s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
a24=mod("d24a_trace",HERE.parent/"run_phase2_d24a_native.py");N=100;DT=.02
def maxrun(x):
 cur=best=0
 for v in x:cur=cur+1 if v else 0;best=max(best,cur)
 return best
def main():
 p=argparse.ArgumentParser();add_launcher_args(p);a,h=setup_preset_cli(p);sys.argv=[sys.argv[0],*h];cfg,agent=resolve_task_config("Isaac-Exp012-G1-Reverse-PhaseR1-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=N;cfg.episode_length_s=10.;cfg.seed=20269031
 if a.device:cfg.sim.device=agent.device=a.device
 with launch_simulation(cfg,a):
  w=RslRlVecEnvWrapper(gym.make("Isaac-Exp012-G1-Reverse-PhaseR1-v0",cfg=cfg),clip_actions=agent.clip_actions);env=w.unwrapped;dev=env.device;policy=a24.Policy(a24.CKPT).to(dev).eval();robot=env.scene["robot"];term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True;sensor=env.scene.sensors["contact_forces"];feet=[i for i,n in enumerate(sensor.body_names) if "ankle_roll" in n];rf=[next(i for i,n in enumerate(robot.body_names) if n==sensor.body_names[j]) for j in feet];obs,_=w.reset();obs=obs["policy"].to(dev)
  arr={k:[] for k in ("applied_torque","computed_torque","effort_limit","torque_ratio","joint_velocity","velocity_limit","contact_force","contact_history","previous_action","yaw","vx","vy","command","fall","slip","impact","velocity_saturation")};fall=torch.zeros(N,dtype=torch.bool,device=dev);slip=fall.clone();impact=fall.clone();vfail=fall.clone();ss=torch.zeros(N,dtype=torch.long,device=dev);vs=ss.clone();root0=robot.data.root_pos_w[:,0].clone();first=torch.full((N,),-1,dtype=torch.long,device=dev)
  for step in range(500):
   t=step*DT;speed=torch.tensor(0.,device=dev) if t<1 else .6*a24.mj(torch.tensor(t-1,device=dev)) if t<2 else .6+(.6*a24.mj(torch.tensor(t-2,device=dev))) if t<3 else torch.tensor(1.2,device=dev);term.external_override[:,0]=speed;term.external_override[:,1:]=0
   with torch.inference_mode():act=policy(obs,torch.zeros(N,device=dev))
   obs2,_,done,ex=w.step(act);obs2=obs2["policy"].to(dev);timeout=ex.get("time_outs",torch.zeros_like(done)).bool();fall|=done.bool()&~timeout;force=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);contact=force>5;fv=robot.data.body_lin_vel_w[:,rf,:2].norm(dim=-1);bad=((fv>.55)&contact).any(1);ss=torch.where(bad,ss+1,torch.zeros_like(ss));slip|=ss>=5;impact|=force.amax(1)>3500;lim=robot.data.joint_vel_limits;lim=lim[...,1].abs() if lim.ndim==3 else lim.abs();vr=robot.data.joint_vel.abs()/lim.clamp_min(1e-6);vs=torch.where(vr.amax(1)>.95,vs+1,torch.zeros_like(vs));vfail|=vs>=5;eff=robot.data.joint_effort_limits.abs().clamp_min(1e-6);tr=robot.data.applied_torque.abs()/eff;single=contact.sum(1)==1;new=(first<0)&single&((robot.data.root_pos_w[:,0]-root0)>0)&(step>=50);first[new]=step
   vals={"applied_torque":robot.data.applied_torque,"computed_torque":getattr(robot.data,"computed_torque",robot.data.applied_torque),"effort_limit":eff,"torque_ratio":tr,"joint_velocity":robot.data.joint_vel,"velocity_limit":lim,"contact_force":force,"contact_history":sensor.data.net_forces_w_history[:,:,feet,:],"previous_action":env.action_manager.prev_action,"yaw":robot.data.root_ang_vel_b[:,2],"vx":robot.data.root_lin_vel_b[:,0],"vy":robot.data.root_lin_vel_b[:,1],"command":torch.full((N,),float(speed),device=dev),"fall":fall,"slip":slip,"impact":impact,"velocity_saturation":vfail}
   for k,v in vals.items():arr[k].append(v.detach().cpu().numpy());obs=obs2
  z={k:np.stack(v) for k,v in arr.items()};OUT.mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT/"stage2q_native_full_trace.npz",**z)
  rows=[]
  for i in range(N):
   contact=(z["contact_force"][:,i]>5);flight=contact.sum(1)==0;hist=bool(flight[200:,].mean()<.1);strict=(np.abs(z["vx"][:,i]-.6)<=.15)&(np.abs(z["vy"][:,i])<=.10)&(np.abs(z["yaw"][:,i])<=.12);strict25=maxrun(strict[50:125])>=25;demo=(np.abs(z["vx"][:,i]-z["command"][:,i])<=.15)&(np.abs(z["vy"][:,i])<=.10)&(np.abs(z["yaw"][:,i])<=.12);demo25=maxrun(demo[50:200])>=25;tv=z["torque_ratio"][:,i];dwell=max(maxrun(tv[:,j]>.95) for j in range(tv.shape[1]));safe=not (z["fall"][-1,i] or z["slip"][-1,i] or z["impact"][-1,i] or z["velocity_saturation"][-1,i]);rows.append({"episode":i,"historical_success":hist,"strict_success":bool(strict25 and safe),"E2_safe_demonstration":bool(first[i]>=0 and demo25 and safe),"first_step":int(first[i]),"torque_max":float(tv.max()),"torque_p95":float(np.quantile(tv,.95)),"torque_max_dwell_gt95":int(dwell),"canonical_velocity_saturation":bool(z["velocity_saturation"][-1,i]),"fall":bool(z["fall"][-1,i]),"dangerous_slip":bool(z["slip"][-1,i]),"impact":bool(z["impact"][-1,i])})
  result={"rows":rows,"historical_success_count":sum(r["historical_success"] for r in rows),"strict_success_count":sum(r["strict_success"] for r in rows),"E2_safe_demonstration_count":sum(r["E2_safe_demonstration"] for r in rows),"first_step_count":sum(r["first_step"]>=0 for r in rows),"persistent_update":0};(OUT/"stage2q_native_full_trace_results.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2));w.close()
if __name__=="__main__":main()
