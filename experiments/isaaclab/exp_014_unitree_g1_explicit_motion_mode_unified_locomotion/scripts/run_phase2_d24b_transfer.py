"""D24B train-only Stage2Q transfer and W_MOVE handoff evaluation."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from collections import OrderedDict
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve();REPO=HERE.parents[4]
OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d24b_native_start_contract_and_recovery";RAW=OUT/"raw"
D16=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d16_dedicated_start_specialist"
S2Q=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
DT=.02;N=64;ROUTES=5;TOTAL=N*ROUTES
def mod(name,path):s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d16=mod("d16_d24b",HERE.parent/"run_phase2_d16_train.py");d3,d6=d16.d3,d16.d6
from g1_explicit_motion_mode.contract import MotionMode,minimum_jerk
from g1_omnidirectional.policy import FrozenGaitActor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli

class Stage2Q(nn.Module):
 def __init__(self,path):
  super().__init__();s=torch.load(path,map_location="cpu",weights_only=False)["actor_state_dict"]
  self.bw=nn.Parameter(s["first_base_weight"],requires_grad=False);self.gw=nn.Parameter(s["first_gait_column"],requires_grad=False);self.bb=nn.Parameter(s["first_bias"],requires_grad=False)
  self.hidden=nn.Sequential(nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,37));self.hidden.load_state_dict(OrderedDict((k.removeprefix("hidden."),v) for k,v in s.items() if k.startswith("hidden.")))
 def forward(self,o):return self.hidden(nn.functional.linear(o,self.bw,self.bb))
def f(x):return float(x.detach().cpu())
def sha_tensor(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().contiguous().cpu().numpy().tobytes())
 return h.hexdigest()
def restore(world,pool,valid):
 picks=torch.tensor(valid,dtype=torch.long);idx=picks.repeat(ROUTES);snap={k:v[idx].to(world.device) for k,v in pool["snapshot"].items()};world.restore_snapshot(snap)
 world.state.target_mode.fill_(int(MotionMode.STAND));world.state.previous_target_mode.fill_(int(MotionMode.STAND));world.state.time_since_mode_change_s.zero_();world.state.ramp_progress.fill_(1);world.state.physical_command.zero_();world.state.previous_physical_command.zero_();world.term.external_override[:,:3].zero_();world.term._update_command();world.env.sim.forward()
 return idx
def physical_command(step,dev):
 if step<50:return torch.tensor(0.,device=dev)
 if step<100:return .6*minimum_jerk(torch.tensor((step-50)/50,device=dev))
 if step<150:return torch.tensor(.6,device=dev)
 if step<188:return .6+(.3-.6)*minimum_jerk(torch.tensor((step-150)/38,device=dev))
 return torch.tensor(.3,device=dev)
def main():
 p=argparse.ArgumentParser();add_launcher_args(p);a,h=setup_preset_cli(p);sys.argv=[sys.argv[0],*h]
 cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=TOTAL;cfg.seed=20279852;cfg.episode_length_s=12.;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None
 if a.device:cfg.sim.device=agent.device=a.device
 pool=torch.load(D16/"raw/train_start_snapshots.pt",map_location="cpu",weights_only=False);valid=[i for i,x in enumerate(pool["valid"]) if x][:N];assert len(valid)==N
 RAW.mkdir(parents=True,exist_ok=True)
 with launch_simulation(cfg,a):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=d16.StartWorld(wrapped,d3.load_resets(),pool);dev=world.device
  hold=d3.initialize("P0_STAND_PARENT",dev)[0].eval();stage=Stage2Q(S2Q).to(dev).eval();walk=FrozenGaitActor(d3.WMOVE).to(dev).eval();idx=restore(world,pool,valid)
  route=torch.arange(ROUTES,device=dev).repeat_interleave(N);source=torch.arange(N,device=dev).repeat(ROUTES);route_names=["R0_SHOLD_ONLY","R1_STAGE2Q_EXACT_NATIVE_PREROLL","R2_BLEND4_NATIVE_PREROLL","R3_BLEND8_NATIVE_PREROLL","R4_BLEND12_NATIVE_PREROLL"]
  # Verify 0.5 s S_HOLD validity before START.
  obs=world.obs();validity=torch.ones(TOTAL,dtype=torch.bool,device=dev);prev_action=world.env.action_manager.prev_action.clone()
  for _ in range(25):
   with torch.inference_mode():act=hold.mean(obs)
   _,_,done,ex=wrapped.step(act);obs=world.obs();timeout=ex.get("time_outs",torch.zeros_like(done)).bool();validity&=~(done.bool()&~timeout);validity&=(world.robot.data.root_lin_vel_b[:,:2].norm(dim=1)<=.08)&(world.robot.data.root_ang_vel_b[:,2].abs()<=.08);prev_action=act
  world.state.request(torch.full((TOTAL,),int(MotionMode.WALK),device=dev));world.state.time_since_mode_change_s.zero_();world.state.ramp_progress.zero_();root0=world.robot.data.root_pos_w[:,0].clone()
  flags=[torch.zeros(TOTAL,dtype=torch.bool,device=dev) for _ in range(6)];streaks=[torch.zeros(TOTAL,dtype=torch.long,device=dev) for _ in range(3)]
  first=torch.full((TOTAL,),-1,dtype=torch.long,device=dev);trackst=torch.zeros_like(first);complete=torch.full_like(first,-1);switch_action_l2=torch.full((TOTAL,),float("nan"),device=dev);switch_cos=torch.full_like(switch_action_l2,float("nan"));switch_jump=torch.full_like(switch_action_l2,float("nan"));contact_before=None
  arrays={k:[] for k in ("obs_141","final_action","stage2q_action","root_state","joint_pos","joint_vel","contact","command","torque_ratio")}
  for step in range(288):
   speed=physical_command(step,dev);cmd=torch.zeros(TOTAL,3,device=dev);cmd[:,0]=speed;progress=torch.full((TOTAL,),min(1.,step/50),device=dev);world.state.advance(cmd,progress,0 if step==0 else DT);d6.set_command(world,cmd)
   with torch.inference_mode():ha=hold.mean(obs);sa=stage(obs[:,:123]);wa=walk(obs[:,:123],torch.zeros(TOTAL,device=dev))
   action=sa.clone();action[route==0]=ha[route==0]
   for rid,b in ((2,4),(3,8),(4,12)):
    m=route==rid
    if step<b:
     u=minimum_jerk(torch.tensor(step/max(1,b-1),device=dev));action[m]=(1-u)*ha[m]+u*sa[m]
   # Only demonstration-success candidates enter the formal speed bridge / W_MOVE handoff.
   demo=(complete>=0)&(complete<=149)&validity&~torch.stack(flags).any(0)&(route>0)
   if step>=213:
    if step==213:
     switch_action_l2[demo]=(wa[demo]-action[demo]).norm(dim=1);switch_cos[demo]=nn.functional.cosine_similarity(wa[demo],action[demo],dim=1);switch_jump[demo]=(wa[demo]-action[demo]).norm(dim=1);contact_before=world.sensor.data.net_forces_w_history[:,-1,world.sf,:].norm(dim=-1)>5
    action[demo]=wa[demo]
   obs_before=obs
   _,_,done,ex=wrapped.step(action);obs=world.obs();timeout=ex.get("time_outs",torch.zeros_like(done)).bool();flags[0]|=done.bool()&~timeout
   force=world.sensor.data.net_forces_w_history[:,-1,world.sf,:].norm(dim=-1);contact=force>5;fv=world.robot.data.body_lin_vel_w[:,world.rf,:2].norm(dim=-1);bad=((fv>.55)&contact).any(1);streaks[0]=torch.where(bad,streaks[0]+1,torch.zeros_like(streaks[0]));flags[1]|=streaks[0]>=5;flags[2]|=force.amax(1)>3500
   lim=world.limits;vr=world.robot.data.joint_vel.abs().div(lim.clamp_min(1e-6)).amax(1);eff=world.robot.data.joint_effort_limits.abs().clamp_min(1e-6);trj=world.robot.data.applied_torque.abs().div(eff);tr=trj.amax(1);streaks[1]=torch.where(vr>.95,streaks[1]+1,torch.zeros_like(streaks[1]));streaks[2]=torch.where(tr>.95,streaks[2]+1,torch.zeros_like(streaks[2]));flags[3]|=streaks[1]>5;flags[4]|=streaks[2]>5;flags[5]|=~torch.isfinite(world.robot.data.root_state_w).all(1)
   single=contact.sum(1)==1;new=(first<0)&single&((world.robot.data.root_pos_w[:,0]-root0)>0);first[new]=step
   target=.3 if step>=188 else .6;good=(world.robot.data.root_lin_vel_b[:,0]-target).abs()<=.15;good&=world.robot.data.root_lin_vel_b[:,1].abs()<=.10;good&=world.robot.data.root_ang_vel_b[:,2].abs()<=.12;trackst=torch.where(good,trackst+1,torch.zeros_like(trackst));newc=(complete<0)&(trackst>=25)&(step<150);complete[newc]=step
   for k,v in {"obs_141":obs_before,"final_action":action,"stage2q_action":sa,"root_state":world.robot.data.root_state_w,"joint_pos":world.robot.data.joint_pos,"joint_vel":world.robot.data.joint_vel,"contact":contact,"command":cmd,"torque_ratio":trj}.items():arrays[k].append(v.detach().cpu().numpy())
   prev_action=action
  flags_cpu=[x.cpu() for x in flags];rows=[]
  for i in range(TOTAL):
   strict=bool(complete[i]>=0 and complete[i]<=74 and validity[i] and not any(x[i] for x in flags));demo=bool(complete[i]>=0 and complete[i]<=149 and validity[i] and not any(x[i] for x in flags));ret=False
   if demo and int(route[i])>0:
    vx=np.stack(arrays["root_state"])[213:288,i,7];vy=np.stack(arrays["root_state"])[213:288,i,8];yaw=np.stack(arrays["root_state"])[213:288,i,12];ret=bool(np.mean(np.abs(vx-.3))<=.15 and np.mean(np.abs(vy))<=.10 and np.mean(np.abs(yaw))<=.12 and not any(x[i] for x in flags))
   rows.append({"episode_id":i,"route":route_names[int(route[i])],"source_index":int(source[i]),"source_pool_index":int(idx[i]),"source_valid":bool(validity[i]),"first_step":bool(first[i]>=0),"first_step_step":int(first[i]),"confirmation_end_step":int(complete[i]),"strict_success":strict,"demonstration_success":demo,"wmove_retained":ret,"fall":bool(flags[0][i]),"dangerous_slip":bool(flags[1][i]),"impact":bool(flags[2][i]),"velocity_long_dwell":bool(flags[3][i]),"torque_long_dwell":bool(flags[4][i]),"nonfinite":bool(flags[5][i]),"handoff_action_l2":None if torch.isnan(switch_action_l2[i]) else f(switch_action_l2[i]),"handoff_action_cosine":None if torch.isnan(switch_cos[i]) else f(switch_cos[i]),"joint_target_jump":None if torch.isnan(switch_jump[i]) else f(switch_jump[i])})
  arr={k:np.stack(v) for k,v in arrays.items()};np.savez_compressed(RAW/"transfer_trajectories.npz",**arr)
  out={"rows":rows,"route_names":route_names,"source_pool_indices":valid,"tensor_hashes_before":{"S_HOLD":sha_tensor(hold),"Stage2Q":sha_tensor(stage),"W_MOVE":sha_tensor(walk)},"tensor_hashes_after":{"S_HOLD":sha_tensor(hold),"Stage2Q":sha_tensor(stage),"W_MOVE":sha_tensor(walk)},"persistent_update":0,"validation_access":0,"heldout_access":0};(RAW/"transfer_results.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({r:{"strict":sum(x["strict_success"] for x in rows if x["route"]==r),"demo":sum(x["demonstration_success"] for x in rows if x["route"]==r),"retain":sum(x["wmove_retained"] for x in rows if x["route"]==r)} for r in route_names},indent=2));wrapped.close()
if __name__=="__main__":main()
