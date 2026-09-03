"""Read-only exp012 Stage 2Q native forward START reproduction for D24A."""
from __future__ import annotations
import argparse, hashlib, json, sys
from collections import OrderedDict
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve();REPO=HERE.parents[4];EXP12=REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion";sys.path.insert(0,str(EXP12/"src"));sys.path.insert(0,str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"));OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d24a_existing_start_teacher_transfer";RAW=OUT/"raw";CKPT=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt";DT=.02;N=100
import isaaclab_tasks,g1_single_policy.tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
def mj(x):x=torch.clamp(x,0,1);return 10*x**3-15*x**4+6*x**5
def ah(*xs):
 h=hashlib.sha256()
 for x in xs:h.update(x.detach().contiguous().cpu().numpy().tobytes())
 return h.hexdigest()
class Policy(nn.Module):
 def __init__(self,path):
  super().__init__();s=torch.load(path,map_location="cpu",weights_only=False)["actor_state_dict"];self.bw=nn.Parameter(s["first_base_weight"],requires_grad=False);self.gw=nn.Parameter(s["first_gait_column"],requires_grad=False);self.bb=nn.Parameter(s["first_bias"],requires_grad=False);self.hidden=nn.Sequential(nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,37));self.hidden.load_state_dict(OrderedDict((k.removeprefix("hidden."),v) for k,v in s.items() if k.startswith("hidden.")))
 def forward(self,o,g):return self.hidden(nn.functional.linear(o,self.bw,self.bb)+g[:,None]*self.gw.T)
def main():
 p=argparse.ArgumentParser();add_launcher_args(p);a,h=setup_preset_cli(p);sys.argv=[sys.argv[0],*h];cfg,agent=resolve_task_config("Isaac-Exp012-G1-Reverse-PhaseR1-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=N;cfg.episode_length_s=6.;cfg.seed=20269031
 if a.device:cfg.sim.device=agent.device=a.device
 with launch_simulation(cfg,a):
  w=RslRlVecEnvWrapper(gym.make("Isaac-Exp012-G1-Reverse-PhaseR1-v0",cfg=cfg),clip_actions=agent.clip_actions);env=w.unwrapped;dev=env.device;policy=Policy(CKPT).to(dev).eval();robot=env.scene["robot"];term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True;sensor=env.scene.sensors["contact_forces"];feet=[i for i,n in enumerate(sensor.body_names) if "ankle_roll" in n];rf=[next(i for i,n in enumerate(robot.body_names) if n==sensor.body_names[j]) for j in feet];obs,_=w.reset();obs=obs["policy"].to(dev);source_hash=[ah(obs[i],robot.data.root_state_w[i],robot.data.joint_pos[i],robot.data.joint_vel[i]) for i in range(N)];root0=robot.data.root_pos_w[:N].clone();arr={k:[] for k in ("obs_123","action","root_state","joint_pos","joint_vel","contact","force","foot_pos","foot_vel","vx","vy","yaw","roll_pitch","command")};flags={k:torch.zeros(N,dtype=torch.bool,device=dev) for k in ("fall","slip","impact","velocity","torque","nonfinite")};slipst=torch.zeros(N,dtype=torch.long,device=dev);velst=torch.zeros_like(slipst);torquest=torch.zeros_like(slipst);first=torch.full((N,),-1,dtype=torch.long,device=dev);acq=torch.full_like(first,-1);streak=torch.zeros_like(first);flight=torch.zeros(N,device=dev);walksteps=torch.zeros(N,device=dev)
  for step in range(200):
   t=step*DT;speed=torch.tensor(0.,device=dev) if t<1 else .6*mj(torch.tensor((t-1)/1.,device=dev)) if t<2 else torch.tensor(.6,device=dev);term.external_override[:,0]=speed;term.external_override[:,1:]=0;g=torch.zeros(N,device=dev)
   with torch.inference_mode():act=policy(obs,g)
   obs2,_,done,extras=w.step(act);obs2=obs2["policy"].to(dev);timeout=extras.get("time_outs",torch.zeros_like(done)).bool();flags["fall"]|=done.bool()&~timeout;force=sensor.data.net_forces_w_history[:N,-1,feet,:].norm(dim=-1);contact=force>5;walksteps+=(step>=50);flight+=(contact.sum(1)==0)&(step>=50);fv=robot.data.body_lin_vel_w[:N,rf,:];bad=((fv[:,:,:2].norm(dim=-1)>.55)&contact).any(1);slipst=torch.where(bad,slipst+1,torch.zeros_like(slipst));flags["slip"]|=slipst>=5;flags["impact"]|=force.amax(1)>3500;limits=robot.data.joint_vel_limits[:N];limits=limits[...,1].abs() if limits.ndim==3 else limits.abs();vr=robot.data.joint_vel[:N].abs().div(limits.clamp_min(1e-6)).amax(1);velst=torch.where(vr>.95,velst+1,torch.zeros_like(velst));flags["velocity"]|=velst>=5;eff=robot.data.joint_effort_limits[:N].abs().clamp_min(1e-6);tr=robot.data.applied_torque[:N].abs().div(eff).amax(1);torquest=torch.where(tr>.95,torquest+1,torch.zeros_like(torquest));flags["torque"]|=torquest>=5;flags["nonfinite"]|=~torch.isfinite(robot.data.root_state_w[:N]).all(1)
   vx=robot.data.root_lin_vel_b[:N,0];vy=robot.data.root_lin_vel_b[:N,1];yaw=robot.data.root_ang_vel_b[:N,2];single=contact.sum(1)==1;disp=robot.data.root_pos_w[:N,0]-root0[:,0];new=(first<0)&single&(disp>0)&(step>=50);first[new]=step;good=(vx-.6).abs()<=.15;good&=vy.abs()<=.10;streak=torch.where(good,streak+1,torch.zeros_like(streak));newa=(acq<0)&(streak>=25)&(step<150);acq[newa]=step
   vals={"obs_123":obs,"action":act,"root_state":robot.data.root_state_w[:N],"joint_pos":robot.data.joint_pos[:N],"joint_vel":robot.data.joint_vel[:N],"contact":contact,"force":force,"foot_pos":robot.data.body_pos_w[:N,rf,:],"foot_vel":fv,"vx":vx,"vy":vy,"yaw":yaw,"roll_pitch":robot.data.projected_gravity_b[:N,:2],"command":torch.full((N,),float(speed),device=dev)}
   for k,v in vals.items():arr[k].append(v.detach().cpu().numpy());obs=obs2
  native_acquired=flight/walksteps.clamp_min(1)<.10;success=(acq>=0)&native_acquired&~flags["fall"]&~flags["slip"]&~flags["torque"]&~flags["nonfinite"];arrays={k:np.stack(v) for k,v in arr.items()};RAW.mkdir(parents=True,exist_ok=True);np.savez_compressed(RAW/"stage2q_native_trajectories.npz",**arrays)
  rows=[]
  for i in range(N):rows.append({"episode":i,"source_hash":source_hash[i],"first_step":int(first[i]),"acquisition_step":int(acq[i]),"native_gait_acquired":bool(native_acquired[i]),"success":bool(success[i]),**{k:bool(v[i]) for k,v in flags.items()}})
  result={"checkpoint":str(CKPT.relative_to(REPO)),"checkpoint_sha256":hashlib.sha256(CKPT.read_bytes()).hexdigest(),"episodes":N,"seed":cfg.seed,"rows":rows,"native_gait_acquisition_rate":float(native_acquired.float().mean()),"success_rate":float(success.float().mean()),"confirmation_rate":float((acq>=0).float().mean()),"fall_rate":float(flags["fall"].float().mean()),"dangerous_slip_rate":float(flags["slip"].float().mean()),"velocity_saturation_rate":float(flags["velocity"].float().mean()),"torque_saturation_rate":float(flags["torque"].float().mean()),"gate":bool(success.float().mean()>=.9 and (acq>=0).float().mean()>=.9 and flags["fall"].float().mean()<=.05 and flags["slip"].float().mean()<=.10 and flags["torque"].float().mean()<=.10),"source_lifecycle":"native exp012 reset (official seed 20269031); 1.0 s zero command; 1.0 s minimum-jerk 0->0.6; hold 0.6","persistent_update":0,"validation_access":0};(RAW/"native_results.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({k:result[k] for k in ("native_gait_acquisition_rate","success_rate","confirmation_rate","fall_rate","dangerous_slip_rate","velocity_saturation_rate","torque_saturation_rate","gate")},indent=2))
if __name__=="__main__":main()
