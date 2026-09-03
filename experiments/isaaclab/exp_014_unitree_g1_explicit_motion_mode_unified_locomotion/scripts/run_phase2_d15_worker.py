"""Read-only Isaac worker for D15 STAND-to-OMNI-WALK start evaluation."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, sys
from pathlib import Path
import gymnasium as gym
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]; DT=.02
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
d3=module("d3_d15",HERE.parent/"run_phase2_d3.py"); d6=module("d6_d15",HERE.parent/"run_phase2_d6_audit.py")
from g1_explicit_motion_mode.contract import MotionMode,minimum_jerk
from g1_omnidirectional.policy import FrozenGaitActor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli

def sha_tensor(*xs):
 h=hashlib.sha256()
 for x in xs:h.update(x.detach().contiguous().cpu().numpy().tobytes())
 return h.hexdigest()
def emit(kind,value,ack=False):
 print("D15_IPC:"+json.dumps({"kind":kind,"value":value},sort_keys=True,separators=(",",":")),flush=True)
 if ack and sys.stdin.readline().strip()!="D15_ACK":raise RuntimeError("parent durability acknowledgement missing")
def conditions():
 x=[{"condition_id":i,"kind":"zero_yaw","direction_deg":22.5*i,"speed":.3,"yaw":0.} for i in range(16)]
 for d in range(8):
  for y in (-.3,.3):x.append({"condition_id":len(x),"kind":"moving_yaw","direction_deg":45.*d,"speed":.3,"yaw":y})
 for y in (-.3,.3):x.append({"condition_id":len(x),"kind":"pure_yaw","direction_deg":0.,"speed":0.,"yaw":y})
 return x
def target_matrix(spec,n,device):
 t=torch.zeros(n,3,device=device); a=math.radians(spec["direction_deg"]); t[:,0]=spec["speed"]*math.cos(a);t[:,1]=spec["speed"]*math.sin(a);t[:,2]=spec["yaw"];return t
def safety(world,n,done,extras,streaks):
 timeout=extras.get("time_outs",torch.zeros_like(done)).bool(); fall=done[:n].bool()&~timeout[:n]
 force=world.sensor.data.net_forces_w_history[:n,-1,world.sf,:].norm(dim=-1); contact=force>5
 feet=world.robot.data.body_lin_vel_w[:n,world.rf,:2].norm(dim=-1); bad=((feet>.55)&contact).any(1)
 streaks[0]=torch.where(bad,streaks[0]+1,torch.zeros_like(streaks[0])); slip=streaks[0]>=5; impact=force.amax(1)>3500
 vr=world.robot.data.joint_vel[:n].abs().div(world.limits[:n].clamp_min(1e-6)).amax(1)
 effort=world.robot.data.joint_effort_limits[:n].abs().clamp_min(1e-6);tr=world.robot.data.applied_torque[:n].abs().div(effort).amax(1)
 streaks[1]=torch.where(vr>.95,streaks[1]+1,torch.zeros_like(streaks[1]));streaks[2]=torch.where(tr>.95,streaks[2]+1,torch.zeros_like(streaks[2]))
 finite=torch.isfinite(world.robot.data.root_state_w[:n]).all(1)&torch.isfinite(world.robot.data.joint_pos[:n]).all(1)
 return fall,slip,impact,streaks[1]>=5,streaks[2]>=5,~finite,contact,force

def stand_snapshots(world,hold,recipes):
 n=len(recipes);pad=recipes+[recipes[-1]]*(world.env.num_envs-n);obs=world.restore(torch.tensor(pad,device=world.device));dev=world.device
 first=torch.full((n,),-1,dtype=torch.long,device=dev);streak=torch.zeros(n,dtype=torch.long,device=dev);complete=torch.full_like(first,-1);captured=torch.zeros(n,dtype=torch.bool,device=dev)
 fall=torch.zeros(n,dtype=torch.bool,device=dev);slip=fall.clone();impact=fall.clone();streaks=[torch.zeros(n,dtype=torch.long,device=dev) for _ in range(3)]
 template=world.snapshot();snap={k:torch.empty_like(v) for k,v in template.items()}; pre_actions=torch.zeros(n,10,37); actions=[]
 for step in range(240):
  with torch.inference_mode():action=hold.mean(obs)
  actions.append(action[:n].detach().cpu());obs,_,done,extras=world.step(action,None)
  f,s,i,_,_,nf,_,_=safety(world,n,done,extras,streaks);fall|=f|nf;slip|=s;impact|=i
  speed=world.robot.data.root_lin_vel_b[:n,:2].norm(dim=1);yaw=world.robot.data.root_ang_vel_b[:n,2].abs();good=(speed<=.08)&(yaw<=.08)
  enter=(first<0)&good;first[enter]=step;streak=torch.where(good,streak+1,torch.zeros_like(streak))
  new=(complete<0)&(streak>=50)&(first<=50);complete[new]=step
  due=(~captured)&(complete>=0)&(step==complete+50)
  if due.any():
   current=world.snapshot()
   due_ids=due.nonzero().flatten()
   for k in snap:snap[k][due_ids]=current[k][due_ids]
   for j in due_ids.tolist():pre_actions[j]=torch.stack(actions[-10:])[:,j]
   captured|=due
 valid=captured&~fall&~slip&~impact
 # Invalid rows receive the final state solely so paired tensor shapes remain fixed.
 current=world.snapshot();missing=~captured;missing_ids=missing.nonzero().flatten()
 for k in snap:snap[k][missing_ids]=current[k][missing_ids]
 for j in missing_ids.tolist():pre_actions[j]=torch.stack(actions[-10:])[:,j]
 # The simulator has 128 envs while the fixed validation cohort has 102.
 # Keep padding deterministic and finite; padding never contributes a result.
 for k in snap:
  if snap[k].shape[0]>n:snap[k][n:]=snap[k][n-1:n]
 hashes=[sha_tensor(*(snap[k][j:j+1] for k in sorted(snap))) for j in range(n)]
 return {"snapshot":{k:v.cpu() for k,v in snap.items()},"valid":valid.cpu().tolist(),"first":first.cpu().tolist(),"complete":complete.cpu().tolist(),"fall":fall.cpu().tolist(),"slip":slip.cpu().tolist(),"impact":impact.cpu().tolist(),"hashes":hashes,"pre_actions":pre_actions}

def evaluate_condition(world,walk,base,spec,episode_ids):
 world.restore_snapshot({k:v.to(world.device) for k,v in base["snapshot"].items()});n=len(episode_ids);dev=world.device;target=target_matrix(spec,world.env.num_envs,dev);world.state.request(torch.full((world.env.num_envs,),int(MotionMode.WALK),device=dev));gait=torch.zeros(world.env.num_envs,device=dev);initial_observation=world.obs()[:n].detach().cpu()
 streak=torch.zeros(n,dtype=torch.long,device=dev);completion=torch.full((n,),-1,dtype=torch.long,device=dev);first=torch.full_like(completion,-1)
 flags=[torch.zeros(n,dtype=torch.bool,device=dev) for _ in range(6)]; streaks=[torch.zeros(n,dtype=torch.long,device=dev) for _ in range(3)]
 vecerr=[];direrr=[];yawerr=[];speed=[];actions=[];root_acc=[];joint_acc=[];torque=[]
 prev_root_v=world.robot.data.root_vel_w[:n].clone();prev_joint_v=world.robot.data.joint_vel[:n].clone();prev_contact=world.sensor.data.net_forces_w_history[:n,-1,world.sf,:].norm(dim=-1)>5
 root_disc=torch.zeros(n,dtype=torch.bool,device=dev);contact_corrupt=root_disc.clone();handoff_failure=root_disc.clone();contact_change=root_disc.clone()
 l2=torch.zeros(n,device=dev);cos=torch.ones(n,device=dev);jump=torch.zeros(n,device=dev)
 for step in range(200):
  p=torch.full((world.env.num_envs,),min(1.,step/25),device=dev);physical=target*minimum_jerk(p)[:,None];world.state.advance(physical,p,0. if step==0 else DT);d6.set_command(world,physical)
  obs124=world.env.observation_manager.compute()["policy"]
  with torch.inference_mode():action=walk(obs124,gait)
  if step==0:
   previous=world.env.action_manager.prev_action[:n].clone();delta=action[:n]-previous;l2=delta.norm(dim=1);cos=F.cosine_similarity(action[:n],previous);jump=(delta*.5).norm(dim=1)
  actions.append(action[:n].detach().cpu());_,_,done,extras=world.wrapped.step(action)
  f,s,i,vs,ts,nf,contact,force=safety(world,n,done,extras,streaks)
  for old,new in zip(flags,(f,s,i,vs,ts,nf)):old|=new
  rv=world.robot.data.root_vel_w[:n];jv=world.robot.data.joint_vel[:n];root_acc.append(((rv-prev_root_v)/DT).norm(dim=1).detach().cpu());joint_acc.append(((jv-prev_joint_v)/DT).norm(dim=1).detach().cpu());torque.append(world.robot.data.applied_torque[:n].norm(dim=1).detach().cpu());prev_root_v=rv.clone();prev_joint_v=jv.clone()
  if step==0:
   root_disc|=~torch.isfinite(rv).all(1);contact_corrupt|=~torch.isfinite(force).all(1);contact_change|=(contact!=prev_contact).any(1);handoff_failure|=f|s|i
  vel=world.robot.data.root_lin_vel_b[:n,:2];yr=world.robot.data.root_ang_vel_b[:n,2]
  ve=(vel-target[:n,:2]).norm(dim=1);sp=vel.norm(dim=1);ye=(yr-target[:n,2]).abs(); dot=(vel*target[:n,:2]).sum(1);cross=vel[:,0]*target[:n,1]-vel[:,1]*target[:n,0];de=torch.rad2deg(torch.atan2(cross.abs(),dot.clamp_min(-1e9))).abs();de=torch.where((sp<1e-6)&(target[:n,:2].norm(dim=1)>0),torch.full_like(de,180.),de)
  pure=spec["kind"]=="pure_yaw";good=(sp<=.08)&(ye<=.08) if pure else (ve<=.12)&(de<=20)&(ye<=.10)
  first[(first<0)&good]=step;streak=torch.where(good,streak+1,torch.zeros_like(streak));new=(completion<0)&(streak>=25)&((step-24)<75);completion[new]=step
  vecerr.append(ve.detach().cpu());direrr.append(de.detach().cpu());yawerr.append(ye.detach().cpu());speed.append(sp.detach().cpu());prev_contact=contact.clone()
 ve=torch.stack(vecerr);de=torch.stack(direrr);ye=torch.stack(yawerr);sp=torch.stack(speed);ra=torch.stack(root_acc);ja=torch.stack(joint_acc);tq=torch.stack(torque);acts=torch.stack(actions);rows=[]
 for j,eid in enumerate(episode_ids):
  comp=int(completion[j]); acquired=comp>=0 and not any(bool(x[j]) for x in flags);hold=False;means={}
  if acquired and comp+76<=200:
   sl=slice(comp+1,comp+76); means={"steady_velocity_error_mean":float(ve[sl,j].mean()),"steady_direction_error_p95_deg":float(torch.quantile(de[sl,j],.95)),"steady_yaw_error_mean":float(ye[sl,j].mean()),"steady_xy_speed_mean":float(sp[sl,j].mean())}
   hold=(means["steady_xy_speed_mean"]<=.08 and means["steady_yaw_error_mean"]<=.08) if spec["kind"]=="pure_yaw" else (means["steady_velocity_error_mean"]<=.12 and means["steady_direction_error_p95_deg"]<=25 and means["steady_yaw_error_mean"]<=.10)
   hold=bool(hold and not any(bool(x[j]) for x in flags))
  valid=bool(base["valid"][j]); joint=bool(acquired and hold)
  if flags[5][j]:primary="NON_FINITE"
  elif not valid:primary="STAND_START_INVALID"
  elif root_disc[j] or contact_corrupt[j] or handoff_failure[j]:primary="HANDOFF_PHYSICAL_FAILURE"
  elif flags[0][j]:primary="FALL_DURING_START"
  elif flags[1][j]:primary="SLIP_DURING_START"
  elif flags[2][j]:primary="IMPACT_FAILURE"
  elif flags[3][j] or flags[4][j]:primary="SATURATION_FAILURE"
  elif comp<0:
   primary="YAW_ACQUISITION_FAILURE" if float(ye[:75,j].mean())>(.08 if spec["kind"]=="pure_yaw" else .10) else "DIRECTION_ACQUISITION_FAILURE" if spec["kind"]!="pure_yaw" and float(de[:75,j].mean())>20 else "WALK_ACQUISITION_TIMEOUT"
  elif not hold:primary="WALK_STEADY_HOLD_FAILURE"
  else:primary="PASS"
  def ff(value):
   value=float(value);return value if math.isfinite(value) else None
  means={k:ff(v) for k,v in means.items()}
  rows.append({"episode_id":eid,"condition_id":spec["condition_id"],"kind":spec["kind"],"direction_deg":spec["direction_deg"],"target_speed":spec["speed"],"target_yaw":spec["yaw"],"recipe_id":d3.VALIDATION[j],"stand_snapshot_hash":base["hashes"][j],"stand_start_valid":valid,"walk_acquisition":bool(acquired),"acquisition_step":None if comp<0 else comp-24,"acquisition_time_s":None if comp<0 else (comp-24)*DT,"walk_steady_hold":hold if acquired else None,"joint_success":joint,"end_to_end_success":bool(valid and joint),"primary_failure":primary,"fall":bool(flags[0][j]),"dangerous_slip":bool(flags[1][j]),"impact":bool(flags[2][j]),"velocity_saturation":bool(flags[3][j]),"torque_saturation":bool(flags[4][j]),"nan_inf":bool(flags[5][j]),"handoff_action_l2":ff(l2[j]),"handoff_action_cosine":ff(cos[j]),"joint_target_jump_rad_l2":ff(jump[j]),"root_state_discontinuity":bool(root_disc[j]),"contact_buffer_corruption":bool(contact_corrupt[j]),"handoff_new_safety_failure":bool(handoff_failure[j]),"contact_state_changed":bool(contact_change[j]),"root_acceleration_p95":ff(torch.quantile(ra[:21,j],.95)),"joint_acceleration_p95":ff(torch.quantile(ja[:21,j],.95)),"actuator_torque_p95":ff(torch.quantile(tq[:21,j],.95)),"observation_hash":sha_tensor(torch.tensor([spec["condition_id"]]),initial_observation[j]),"action_hash":sha_tensor(acts[:,j]),**means})
 return rows

def main():
 p=argparse.ArgumentParser();add_launcher_args(p);args,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0],*hydra]
 cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=128;cfg.seed=20279215;cfg.episode_length_s=20.;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None
 if args.device:cfg.sim.device=agent.device=args.device
 context=False
 with launch_simulation(cfg,args):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=d3.StandWorld(wrapped,d3.load_resets(),torch.zeros(680));hold=d3.initialize("P0_STAND_PARENT",world.device)[0].eval();walk=FrozenGaitActor(d3.WMOVE).to(world.device).eval();base=stand_snapshots(world,hold,d3.VALIDATION)
  emit("SNAPSHOTS",{"recipes":d3.VALIDATION,"valid":base["valid"],"first_entry":base["first"],"hold_completion":base["complete"],"fall":base["fall"],"slip":base["slip"],"impact":base["impact"],"hashes":base["hashes"]},True)
  requested=set(json.loads(os.environ.get("D15_EPISODE_IDS_JSON","[]")))
  for spec in conditions():
   ids=[f"D15-F-{spec['condition_id']:02d}-{j:03d}" for j in range(len(d3.VALIDATION))]
   if requested and not any(eid in requested for eid in ids):continue
   if requested and not all(eid in requested for eid in ids):raise RuntimeError("partial condition resume violates paired cohort contract")
   emit("START_REQUEST",{"episode_ids":ids},True)
   for row in evaluate_condition(world,walk,base,spec,ids):emit("RESULT",row,True)
  wrapped.close();context=True
 emit("WORKER_FINISHED",{"episodes":len(d3.VALIDATION)*34,"simulation_context_teardown":"PASS" if context else "FAIL"})
if __name__=="__main__":main()
