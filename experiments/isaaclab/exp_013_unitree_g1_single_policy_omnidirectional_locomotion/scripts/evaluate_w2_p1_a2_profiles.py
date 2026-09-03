"""Matched-state A2 joint, duration, timing, and command-onset diagnostics."""
from __future__ import annotations
import argparse,csv,json,math,sys
from collections import defaultdict
from pathlib import Path
import gymnasium as gym
import torch

HERE=Path(__file__).resolve();REPO=HERE.parents[4];EXP=HERE.parent.parent
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion";OUT=BASE/"phase_w2_p1_a2_start_boundary_physical_diagnosis"
SELECTED=BASE/"phase_w2_p1_r2_long_horizon_group_balanced_stop_integration/raw/selected_student.pt";PARENT=BASE/"phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt";TEACHER=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
sys.path[:0]=[str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),str(EXP/"src"),str(HERE.parent)]
import isaaclab_tasks  # noqa:F401
import g1_omnidirectional.tasks  # noqa:F401
from g1_omnidirectional.policy import FrozenGaitActor
from g1_omnidirectional.yaw_calibration import calibrate_yaw
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
from train_w2_p1_student import Student

parser=argparse.ArgumentParser();parser.add_argument("--max-envs",type=int,default=4200);add_launcher_args(parser);args,hydra=setup_preset_cli(parser);sys.argv=[sys.argv[0],*hydra]
PROFILES=("J0_STUDENT_ALL","J1_PARENT_LOWER_STUDENT_UPPER","J2_STUDENT_LOWER_PARENT_UPPER","J3_W1B_LOWER_STUDENT_UPPER","J4_STUDENT_LOWER_W1B_UPPER","J5_STOP_LOWER_STUDENT_UPPER","J6_STUDENT_LOWER_STOP_UPPER","D0_SKIP_ZERO","STUDENT_D2","STUDENT_D4","PARENT_D2","PARENT_D4","STOP_D2","STOP_D4","T4_HOLD_PREVIOUS_ACTION_AT_ZERO","ONSET_0P001","ONSET_0P0025","ONSET_0P005","ONSET_0P01","ONSET_0P025","ONSET_0P05")
LOWER=(0,1,3,4,7,8,11,12,15,16,19,20);WAIST=(2,);UPPER=tuple(i for i in range(37) if i not in LOWER+WAIST)
ONSET={"ONSET_0P001":.001,"ONSET_0P0025":.0025,"ONSET_0P005":.005,"ONSET_0P01":.01,"ONSET_0P025":.025,"ONSET_0P05":.05}
def minjerk(x):x=x.clamp(0,1);return x**3*(10-15*x+6*x*x)
def write_csv(path,rows):
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def clone(env,robot,term,n,p):
 origins=env.scene.env_origins;refs=torch.arange(n,device=env.device);targets=torch.arange(p*n,device=env.device);src=targets.remainder(n);local=robot.data.root_pos_w[refs]-origins[refs];pose=torch.cat((local[src]+origins[targets],robot.data.root_quat_w[refs][src]),1);vel=torch.cat((robot.data.root_lin_vel_w[refs][src],robot.data.root_ang_vel_w[refs][src]),1)
 robot.write_root_pose_to_sim(pose,targets);robot.write_root_velocity_to_sim(vel,targets);robot.write_joint_state_to_sim(robot.data.joint_pos[refs][src],robot.data.joint_vel[refs][src],env_ids=targets)
 for _,v in vars(term).items():
  if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==env.num_envs:v[targets]=v[refs][src].clone()
 env.action_manager._action[targets]=env.action_manager.action[refs][src];env.action_manager._prev_action[targets]=env.action_manager.prev_action[refs][src];env.episode_length_buf[targets]=env.episode_length_buf[refs][src];env.sim.forward()
def aggregate(rows,profiles):
 out=[]
 for p in profiles:
  s=[r for r in rows if r["profile"]==p]
  for (d,y),q in sorted(((d,y),[r for r in s if r["direction"]==d and r["yaw"]==y]) for d in range(0,360,45) for y in (-.3,0.,.3)):
   out.append({"profile":p,"direction":d,"yaw":y,"episodes":len(q),"endpoint_success":sum(r["endpoint_success"] for r in q)/len(q),"acquisition_success":sum(r["acquisition_success"] for r in q)/len(q),"fall_rate":sum(r["fall"] for r in q)/len(q),"dangerous_slip_rate":sum(r["dangerous_slip"] for r in q)/len(q),"impact_rate":sum(r["impact"] for r in q)/len(q),"first_movement_median_s":float(torch.tensor([r["first_movement_s"] for r in q if r["first_movement_s"] is not None]).median()) if any(r["first_movement_s"] is not None for r in q) else None,"action_jump_l2":sum(r["action_jump_l2"] for r in q)/len(q),"base_angular_impulse":sum(r["base_angular_impulse"] for r in q)/len(q),"contact_impulse":sum(r["contact_impulse"] for r in q)/len(q)})
 return out
def hybrid(base,source,indices):
 z=base.clone();z[:,list(indices)]=source[:,list(indices)];return z
def main():
 jobs=[(d,y,e) for d in range(0,360,45) for y in (-.3,0.,.3) for e in range(200)];p=len(PROFILES);cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=min(args.max_envs,p*len(jobs));cfg.episode_length_s=12.;cfg.seed=20279001;cfg.observations.policy.enable_corruption=False
 if args.device:cfg.sim.device=agent.device=args.device
 rows=[];traces=[]
 with launch_simulation(cfg,args):
  w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);env=w.unwrapped;dev=env.device;robot=env.scene["robot"];sensor=env.scene["contact_forces"];feet=sensor.find_bodies(".*_ankle_roll_link")[0];rfeet=robot.find_bodies(".*_ankle_roll_link")[0];parent=FrozenGaitActor(PARENT).to(dev).eval();teacher=FrozenGaitActor(TEACHER).to(dev).eval();student=Student(torch.load(SELECTED,map_location="cpu",weights_only=False)["actor_state_dict"]).to(dev).eval();term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True;nmax=env.num_envs//p;cursor=0
  while cursor<len(jobs):
   base=jobs[cursor:cursor+nmax];n=len(base);pad=base+[base[i%n] for i in range(nmax-n)] if n<nmax else base;ids=torch.arange(p*nmax,device=dev);env.reset(env_ids=ids);term.external_override.zero_();term._update_command();obs=w.get_observations().to(dev);gait=torch.zeros(env.num_envs,device=dev)
   for _ in range(round(3/env.step_dt)):
    with torch.inference_mode():a=teacher(obs["policy"],gait)
    obs,_,_,_=w.step(a);obs=obs.to(dev)
   clone(env,robot,term,nmax,p);term.external_override.zero_();term._update_command();obs=w.get_observations().to(dev);prev=env.action_manager.prev_action.clone();force0=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);contact0=force0>5;phase0=torch.where(contact0.all(1),2,torch.where(contact0[:,0]&~contact0[:,1],0,torch.where(contact0[:,1]&~contact0[:,0],1,torch.where(~contact0.any(1),3,4))))
   with torch.inference_mode():sa=student(obs["policy"],gait);pa=parent(obs["policy"],gait);ta=teacher(obs["policy"],gait)
   actions=sa.clone()
   for bi,name in enumerate(PROFILES):
    sl=slice(bi*nmax,(bi+1)*nmax)
    if name in ("J1_PARENT_LOWER_STUDENT_UPPER","J3_W1B_LOWER_STUDENT_UPPER"):actions[sl]=hybrid(sa[sl],pa[sl],LOWER+WAIST)
    elif name in ("J2_STUDENT_LOWER_PARENT_UPPER","J4_STUDENT_LOWER_W1B_UPPER"):actions[sl]=hybrid(sa[sl],pa[sl],UPPER)
    elif name=="J5_STOP_LOWER_STUDENT_UPPER":actions[sl]=hybrid(sa[sl],ta[sl],LOWER+WAIST)
    elif name=="J6_STUDENT_LOWER_STOP_UPPER":actions[sl]=hybrid(sa[sl],ta[sl],UPPER)
    elif name.startswith("PARENT_"):actions[sl]=pa[sl]
    elif name.startswith("STOP_") or name.startswith("ONSET_"):actions[sl]=ta[sl]
    elif name=="T4_HOLD_PREVIOUS_ACTION_AT_ZERO":actions[sl]=prev[sl]
   dirs=torch.tensor([math.radians(x[0]) for x in pad],device=dev);yb=torch.tensor([x[1] for x in pad],device=dev);target=torch.stack((.3*dirs.cos(),.3*dirs.sin(),yb),1).repeat(p,1)
   # D0 has no zero-command action: its first step already receives the first non-zero command.
   d0=PROFILES.index("D0_SKIP_ZERO");sl=slice(d0*nmax,(d0+1)*nmax);alpha=minjerk(torch.tensor(env.step_dt/1.5,device=dev));physical=target[sl]*alpha;term.external_override[sl,:2]=physical[:,:2];term.external_override[sl,2]=calibrate_yaw(physical[:,2]);term._update_command();obs=w.get_observations().to(dev)
   with torch.inference_mode():actions[sl]=student(obs["policy"][sl],gait[sl])
   jump=torch.linalg.vector_norm(actions-prev,dim=1);obs,_,done,extras=w.step(actions);obs=obs.to(dev);fall=done.bool()&~extras.get("time_outs",torch.zeros_like(done)).bool();slip=torch.zeros_like(fall);impact=torch.zeros_like(fall);streak=torch.zeros(env.num_envs,dtype=torch.long,device=dev);first_move=torch.full((env.num_envs,),float("nan"),device=dev);acq=torch.full_like(first_move,float("nan"));sustain=torch.zeros_like(streak);ve_sum=torch.zeros_like(first_move);ye_sum=torch.zeros_like(first_move);measure=torch.zeros_like(first_move);angimp=torch.zeros_like(first_move);contactimp=torch.zeros_like(first_move)
   for k in range(1,round(5.5/env.step_dt)+1):
    t=k*env.step_dt;alpha=minjerk(torch.tensor(t/1.5,device=dev));physical=target*alpha;term.external_override[:,:2]=physical[:,:2];term.external_override[:,2]=calibrate_yaw(physical[:,2]);term._update_command();obs=w.get_observations().to(dev)
    with torch.inference_mode():sact=student(obs["policy"],gait);pact=parent(obs["policy"],gait);tact=teacher(obs["policy"],gait)
    a=sact.clone()
    for bi,name in enumerate(PROFILES):
     sl=slice(bi*nmax,(bi+1)*nmax)
     if name.endswith("_D2") and k<2:a[sl]=pact[sl] if name.startswith("PARENT") else tact[sl] if name.startswith("STOP") else sact[sl]
     elif name.endswith("_D4") and k<4:a[sl]=pact[sl] if name.startswith("PARENT") else tact[sl] if name.startswith("STOP") else sact[sl]
     elif name in ONSET:
      mag=torch.linalg.vector_norm(physical[sl,:2],dim=1);use=mag<ONSET[name];a[sl][use]=tact[sl][use]
    obs,_,dn,ex=w.step(a);obs=obs.to(dev);fall|=dn.bool()&~ex.get("time_outs",torch.zeros_like(dn)).bool();force=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);fs=torch.linalg.vector_norm(robot.data.body_lin_vel_w[:,rfeet,:2],dim=-1);bad=(fs>.55)&(force>5);streak=torch.where(bad.any(1),streak+1,torch.zeros_like(streak));slip|=streak>=5;impact|=force.amax(1)>3500;angimp+=torch.linalg.vector_norm(robot.data.root_ang_vel_b,dim=1)*env.step_dt;contactimp+=force.sum(1)*env.step_dt
    actual=robot.data.root_lin_vel_b[:,:2];ay=robot.data.root_ang_vel_b[:,2];spd=torch.linalg.vector_norm(actual,dim=1);first_move[torch.isnan(first_move)&(spd>.08)]=t;ve=torch.linalg.vector_norm(actual-target[:,:2],dim=1);ye=(ay-target[:,2]).abs();sign=(target[:,2].abs()<1e-8)|(ay*target[:,2]>0);ep=(ve<=.25)&(ye<=.20)&sign&~fall&~slip&~impact
    if t>=1.5:sustain=torch.where(ep,sustain+1,torch.zeros_like(sustain));new=torch.isnan(acq)&(sustain>=round(.2/env.step_dt));acq[new]=t-1.5
    else:sustain.zero_()
    if t>=3.5:ve_sum+=ve;ye_sum+=ye;measure+=1
    if k in (1,2,4,8,16):
     for bi,name in enumerate(PROFILES):
      sl=slice(bi*nmax,(bi+1)*nmax);traces.append({"profile":name,"batch_start":cursor,"steps":k,"mean_base_linear_speed":float(torch.linalg.vector_norm(robot.data.root_lin_vel_b[sl],dim=1).mean()),"mean_base_angular_speed":float(torch.linalg.vector_norm(robot.data.root_ang_vel_b[sl],dim=1).mean()),"mean_height":float(robot.data.root_pos_w[sl,2].mean()),"fall_rate":float(fall[sl].float().mean())})
   endpoint=(ve_sum/measure<=.25)&(ye_sum/measure<=.20)&~fall&~slip&~impact
   for bi,name in enumerate(PROFILES):
    for i,(d,y,e) in enumerate(pad[:n]):
     j=bi*nmax+i;rows.append({"profile":name,"direction":d,"yaw":y,"episode":e,"contact_phase":int(phase0[j]),"endpoint_success":int(endpoint[j]),"acquisition_success":int(torch.isfinite(acq[j]) and acq[j]<=3),"acquisition_time_s":None if torch.isnan(acq[j]) else float(acq[j]),"fall":int(fall[j]),"dangerous_slip":int(slip[j]),"impact":int(impact[j]),"first_movement_s":None if torch.isnan(first_move[j]) else float(first_move[j]),"action_jump_l2":float(jump[j]),"base_angular_impulse":float(angimp[j]),"contact_impulse":float(contactimp[j])})
   cursor+=n;print(json.dumps({"matched_trials":cursor,"total":len(jobs),"profiles":p}),flush=True)
  OUT.mkdir(parents=True,exist_ok=True);write_csv(OUT/"raw_a2_profile_episodes.csv",rows);write_csv(OUT/"raw_a2_profile_traces.csv",traces);summary=aggregate(rows,PROFILES);write_csv(OUT/"raw_a2_profile_summary.csv",summary);(OUT/"raw_a2_profile_summary.json").write_text(json.dumps({"profiles":list(PROFILES),"summary":summary},indent=2)+"\n");w.close()
if __name__=="__main__":main()
