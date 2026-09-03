"""A2 enriched exact reconstruction of the A1/D4 four-branch boundary run."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys
from collections import defaultdict
from pathlib import Path
import gymnasium as gym
import torch

HERE=Path(__file__).resolve();REPO=HERE.parents[4];EXP=HERE.parent.parent
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion";OUT=BASE/"phase_w2_p1_a2_start_boundary_physical_diagnosis";D4=BASE/"phase_w2_p1_d4_heldout_exact_zero_generalization_diagnosis"
SELECTED=BASE/"phase_w2_p1_r2_long_horizon_group_balanced_stop_integration/raw/selected_student.pt";PARENT=BASE/"phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt";TEACHER=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
sys.path.insert(0,str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"));sys.path.insert(0,str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"));sys.path.insert(0,str(EXP/"src"));sys.path.insert(0,str(HERE.parent))
import isaaclab_tasks  # noqa:F401
import g1_omnidirectional.tasks  # noqa:F401
from g1_omnidirectional.policy import FrozenGaitActor
from g1_omnidirectional.yaw_calibration import calibrate_yaw
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
from train_w2_p1_student import Student

parser=argparse.ArgumentParser();parser.add_argument("--max-envs",type=int,default=1600);add_launcher_args(parser);args,hydra=setup_preset_cli(parser);sys.argv=[sys.argv[0],*hydra]
BRANCHES=("B_STUDENT","B_W1B_LABEL","B_STOP_TEACHER","B_CANONICAL_PARENT")
def minjerk(x):x=x.clamp(0,1);return x**3*(10-15*x+6*x*x)
def hrow(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def write_csv(path,rows):
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def clone(env,robot,term,n):
 origins=env.scene.env_origins;refs=torch.arange(n,device=env.device);targets=torch.arange(4*n,device=env.device);src=targets.remainder(n);local=robot.data.root_pos_w[refs]-origins[refs];pose=torch.cat((local[src]+origins[targets],robot.data.root_quat_w[refs][src]),1);vel=torch.cat((robot.data.root_lin_vel_w[refs][src],robot.data.root_ang_vel_w[refs][src]),1)
 robot.write_root_pose_to_sim(pose,targets);robot.write_root_velocity_to_sim(vel,targets);robot.write_joint_state_to_sim(robot.data.joint_pos[refs][src],robot.data.joint_vel[refs][src],env_ids=targets)
 for name,v in vars(term).items():
  if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==env.num_envs:v[targets]=v[refs][src].clone()
 env.action_manager._action[targets]=env.action_manager.action[refs][src];env.action_manager._prev_action[targets]=env.action_manager.prev_action[refs][src];env.episode_length_buf[targets]=env.episode_length_buf[refs][src];env.sim.forward()
def aggregate(rows):
 out=[]
 for branch in BRANCHES:
  s=[r for r in rows if r["branch"]==branch];times=torch.tensor([r["acquisition_time_s"] for r in s if r["acquisition_time_s"] is not None])
  out.append({"branch":branch,"episodes":len(s),"endpoint_success":sum(r["endpoint_success"] for r in s)/len(s),"acquisition_success":sum(r["acquisition_success"] for r in s)/len(s),"acquisition_median_s":float(torch.quantile(times,.5)),"acquisition_p95_s":float(torch.quantile(times,.95)),"fall_rate":sum(r["fall"] for r in s)/len(s),"fall_time_median_s":float(torch.tensor([r["fall_time_s"] for r in s if r["fall_time_s"] is not None]).median()) if any(r["fall_time_s"] is not None for r in s) else None,"dangerous_slip_rate":sum(r["dangerous_slip"] for r in s)/len(s),"impact_rate":sum(r["impact"] for r in s)/len(s),"translation_mae":sum(r["translation_mae"] for r in s)/len(s),"yaw_mae":sum(r["yaw_mae"] for r in s)/len(s)})
 return out
def finalize(rows,state_rows,action_rows,initial_rows):
 OUT.mkdir(parents=True,exist_ok=True);write_csv(OUT/"a1_exact_zero_branch_reconstruction.csv",rows);write_csv(OUT/"start_boundary_state_divergence.csv",state_rows);write_csv(OUT/"start_boundary_action_discontinuity.csv",action_rows);write_csv(OUT/"raw_a2_initial_states.csv",initial_rows)
 summary=aggregate(rows);old=json.loads((D4/"exact_zero_one_step_counterfactual.json").read_text())["summary"];mapping={"B_STUDENT":"A_STUDENT","B_W1B_LABEL":"A_W1B_LABEL","B_STOP_TEACHER":"A_STOP_TEACHER","B_CANONICAL_PARENT":"A_PARENT"};diff=[]
 for x in summary:
  o=next(v for v in old if v["branch"]==mapping[x["branch"]]);diff.extend([abs(x["endpoint_success"]-o["moving_endpoint_success"]),abs(x["acquisition_success"]-o["acquisition_success"]),abs(x["fall_rate"]-o["fall_rate"]),abs(x["dangerous_slip_rate"]-o["dangerous_slip_rate"]),abs(x["impact_rate"]-o["impact_rate"])])
 (OUT/"a1_exact_zero_branch_reconstruction.json").write_text(json.dumps({"summary":summary,"conditions":24,"episodes_per_condition":200,"metric_maximum_difference_from_A1":max(diff),"metric_level_exact":max(diff)<=1e-12,"initial_state_hashes_recorded":True,"previous_action_hashes_recorded":True,"branch_action_hashes_recorded":True},indent=2,sort_keys=True)+"\n")
 (OUT/"start_boundary_state_divergence.json").write_text(json.dumps({"rows":state_rows,"horizons":[1,2,4,8,16],"reference":"B_STUDENT matched state"},indent=2,sort_keys=True)+"\n");(OUT/"start_boundary_action_discontinuity.json").write_text(json.dumps({"rows":action_rows,"action_rate_weight":.005,"branch_and_next_step_saved":True},indent=2,sort_keys=True)+"\n")

def main():
 jobs=[(d,y,e) for d in range(0,360,45) for y in (-.3,0.,.3) for e in range(200)];cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=min(args.max_envs,4*len(jobs));cfg.episode_length_s=12.;cfg.seed=20279001;cfg.observations.policy.enable_corruption=False
 if args.device:cfg.sim.device=agent.device=args.device
 rows=[];state_rows=[];action_rows=[];initial_rows=[]
 with launch_simulation(cfg,args):
  w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);env=w.unwrapped;dev=env.device;robot=env.scene["robot"];sensor=env.scene["contact_forces"];feet=sensor.find_bodies(".*_ankle_roll_link")[0];rfeet=robot.find_bodies(".*_ankle_roll_link")[0];parent=FrozenGaitActor(PARENT).to(dev).eval();teacher=FrozenGaitActor(TEACHER).to(dev).eval();student=Student(torch.load(SELECTED,map_location="cpu",weights_only=False)["actor_state_dict"]).to(dev).eval();term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True;nmax=env.num_envs//4;cursor=0
  while cursor<len(jobs):
   base=jobs[cursor:cursor+nmax];n=len(base);pad=base+[base[i%n] for i in range(nmax-n)] if n<nmax else base;env.reset(env_ids=torch.arange(env.num_envs,device=dev));term.external_override.zero_();term._update_command();obs=w.get_observations().to(dev);gait=torch.zeros(env.num_envs,device=dev)
   for _ in range(round(3/env.step_dt)):
    with torch.inference_mode():a=teacher(obs["policy"],gait)
    obs,_,_,_=w.step(a);obs=obs.to(dev)
   clone(env,robot,term,nmax);term.external_override.zero_();term._update_command();obs=w.get_observations().to(dev);prev=env.action_manager.prev_action.clone();pretorque=robot.data.applied_torque.clone();force0=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);contact0=force0>5;phase0=torch.where(contact0.all(1),2,torch.where(contact0[:,0]&~contact0[:,1],0,torch.where(contact0[:,1]&~contact0[:,0],1,torch.where(~contact0.any(1),3,4))))
   with torch.inference_mode():sa=student(obs["policy"],gait);pa=parent(obs["policy"],gait);ta=teacher(obs["policy"],gait)
   actions=torch.empty_like(sa);actions[:nmax]=sa[:nmax];actions[nmax:2*nmax]=pa[nmax:2*nmax];actions[2*nmax:3*nmax]=ta[2*nmax:3*nmax];actions[3*nmax:]=pa[3*nmax:]
   initial_vector=torch.cat((obs["policy"],robot.data.root_pos_w[:,2:3],force0,contact0.float()),1)
   for bi,bn in enumerate(BRANCHES):
    for i,(d,y,e) in enumerate(pad[:n]):
     j=bi*nmax+i;initial_rows.append({"condition_direction":d,"condition_yaw":y,"episode":e,"branch":bn,"initial_state_hash":hrow(initial_vector[j]),"initial_previous_action_hash":hrow(prev[j]),"branch_action_hash":hrow(actions[j]),"contact_phase":int(phase0[j]),**{f"obs_{q}":float(obs["policy"][j,q]) for q in range(123)}})
   obs,_,done,extras=w.step(actions);obs=obs.to(dev);posttorque=robot.data.applied_torque.clone();next_student=None
   delta=actions-prev;torque_jump=posttorque-pretorque
   fall=done.bool()&~extras.get("time_outs",torch.zeros_like(done)).bool();fall_time=torch.where(fall,torch.zeros(env.num_envs,device=dev),torch.full((env.num_envs,),float("nan"),device=dev));slip=torch.zeros_like(fall);impact=torch.zeros_like(fall);streak=torch.zeros(env.num_envs,dtype=torch.long,device=dev);tiltbad=torch.zeros_like(fall);sat=torch.zeros_like(fall);jointbad=torch.zeros_like(fall);speed_sum=torch.zeros(env.num_envs,device=dev);yaw_sum=torch.zeros_like(speed_sum);ve_sum=torch.zeros_like(speed_sum);ye_sum=torch.zeros_like(speed_sum);measure=torch.zeros_like(speed_sum);first_move=torch.full_like(speed_sum,float("nan"));acq=torch.full_like(speed_sum,float("nan"));sustain=torch.zeros(env.num_envs,dtype=torch.long,device=dev);rollmax=torch.zeros_like(speed_sum);pitchmax=torch.zeros_like(speed_sum);angmax=torch.zeros_like(speed_sum);vzmax=torch.zeros_like(speed_sum);impulse=torch.zeros(env.num_envs,2,device=dev);foot_slip=torch.zeros(env.num_envs,2,device=dev);first_precursor=[None]*env.num_envs
   dirs=torch.tensor([math.radians(x[0]) for x in pad],device=dev);yb=torch.tensor([x[1] for x in pad],device=dev);target=torch.stack((.3*dirs.cos(),.3*dirs.sin(),yb),1).repeat(4,1);horizons={1,2,4,8,16}
   for k in range(1,round(5.5/env.step_dt)+1):
    t=k*env.step_dt;alpha=minjerk(torch.tensor(t/1.5,device=dev));physical=target*alpha;term.external_override[:,:2]=physical[:,:2];term.external_override[:,2]=calibrate_yaw(physical[:,2]);
    if k==1:term._update_command();obs=w.get_observations().to(dev)
    with torch.inference_mode():a=student(obs["policy"],gait)
    if k==1:next_student=a.clone()
    obs,_,dn,ex=w.step(a);obs=obs.to(dev);newfall=dn.bool()&~ex.get("time_outs",torch.zeros_like(dn)).bool()&~fall;fall_time[newfall]=t;fall|=newfall;force=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);fs=torch.linalg.vector_norm(robot.data.body_lin_vel_w[:,rfeet,:2],dim=-1);bad=((fs>.55)&(force>5));streak=torch.where(bad.any(1),streak+1,torch.zeros_like(streak));slip|=streak>=5;impact|=force.amax(1)>3500;impulse+=force*env.step_dt;foot_slip+=bad.float()*env.step_dt
    g=robot.data.projected_gravity_b;roll=torch.atan2(g[:,1].abs(),g[:,2].abs().clamp_min(1e-6));pitch=torch.atan2(g[:,0].abs(),g[:,2].abs().clamp_min(1e-6));rollmax=torch.maximum(rollmax,roll);pitchmax=torch.maximum(pitchmax,pitch);angmax=torch.maximum(angmax,torch.linalg.vector_norm(robot.data.root_ang_vel_b,dim=1));vzmax=torch.maximum(vzmax,robot.data.root_lin_vel_b[:,2].abs());tiltbad|=(roll>.7)|(pitch>.7);sat|=a.abs().amax(1)>=1.0;jointbad|=robot.data.joint_vel.abs().amax(1)>50
    actual=robot.data.root_lin_vel_b[:,:2];ay=robot.data.root_ang_vel_b[:,2];spd=torch.linalg.vector_norm(actual,dim=1);first_move[torch.isnan(first_move)&(spd>.08)]=t;ve=torch.linalg.vector_norm(actual-target[:,:2],dim=1);ye=(ay-target[:,2]).abs();sign=(target[:,2].abs()<1e-8)|(ay*target[:,2]>0);ep=(ve<=.25)&(ye<=.20)&sign&~fall&~slip&~impact
    if t>=1.5:sustain=torch.where(ep,sustain+1,torch.zeros_like(sustain));new=torch.isnan(acq)&(sustain>=round(.2/env.step_dt));acq[new]=t-1.5
    else:sustain.zero_()
    if t>=3.5:speed_sum+=spd;yaw_sum+=ay.abs();ve_sum+=ve;ye_sum+=ye;measure+=1
    if k in horizons:
     state=torch.cat((robot.data.root_lin_vel_b,robot.data.root_ang_vel_b,robot.data.root_pos_w[:,2:3],robot.data.joint_pos,robot.data.joint_vel,contact0.float()),1);ref=state[:nmax]
     for bi,bn in enumerate(BRANCHES):
      cur=state[bi*nmax:(bi+1)*nmax];dist=torch.linalg.vector_norm(cur-ref,dim=1)
      for i,(d,y,e) in enumerate(pad[:n]):state_rows.append({"direction":d,"yaw":y,"episode":e,"branch":bn,"steps":k,"state_l2_from_student":float(dist[i]),"base_linear_speed":float(torch.linalg.vector_norm(robot.data.root_lin_vel_b[bi*nmax+i])),"base_angular_speed":float(torch.linalg.vector_norm(robot.data.root_ang_vel_b[bi*nmax+i])),"base_height":float(robot.data.root_pos_w[bi*nmax+i,2]),"left_contact":int((force[bi*nmax+i,0]>5)),"right_contact":int((force[bi*nmax+i,1]>5)),"left_force":float(force[bi*nmax+i,0]),"right_force":float(force[bi*nmax+i,1])})
   endpoint=(ve_sum/measure<=.25)&(ye_sum/measure<=.20)&~fall&~slip&~impact
   for bi,bn in enumerate(BRANCHES):
    for i,(d,y,e) in enumerate(pad[:n]):
     j=bi*nmax+i;rows.append({"direction":d,"yaw":y,"episode":e,"branch":bn,"endpoint_success":int(endpoint[j]),"acquisition_success":int(torch.isfinite(acq[j]) and acq[j]<=3),"acquisition_time_s":None if torch.isnan(acq[j]) else float(acq[j]),"fall":int(fall[j]),"fall_time_s":None if torch.isnan(fall_time[j]) else float(fall_time[j]),"dangerous_slip":int(slip[j]),"impact":int(impact[j]),"excessive_tilt":int(tiltbad[j]),"action_saturation":int(sat[j]),"joint_limit_proximity":int(jointbad[j]),"translation_mae":float(ve_sum[j]/measure[j]),"yaw_mae":float(ye_sum[j]/measure[j]),"roll_max":float(rollmax[j]),"pitch_max":float(pitchmax[j]),"angular_velocity_max":float(angmax[j]),"vertical_velocity_max":float(vzmax[j]),"left_contact_impulse":float(impulse[j,0]),"right_contact_impulse":float(impulse[j,1]),"left_foot_slip_s":float(foot_slip[j,0]),"right_foot_slip_s":float(foot_slip[j,1]),"contact_phase":int(phase0[j]),"initial_state_hash":hrow(initial_vector[j]),"previous_action_hash":hrow(prev[j]),"branch_action_hash":hrow(actions[j])})
     ar={"direction":d,"yaw":y,"episode":e,"branch":bn,"action_l2":float(torch.linalg.vector_norm(delta[j])),"action_max_abs_delta":float(delta[j].abs().max()),"pd_target_jump_l2":float(torch.linalg.vector_norm(delta[j]*.25)),"estimated_torque_jump_l2":float(torch.linalg.vector_norm(torque_jump[j])),"action_rate_penalty_equivalent":float(.005*delta[j].square().mean()),"next_student_delta_l2":float(torch.linalg.vector_norm(next_student[j]-actions[j]))}
     for q in range(37):ar[f"branch_minus_prev_{q}"]=float(delta[j,q]);ar[f"next_minus_branch_{q}"]=float(next_student[j,q]-actions[j,q])
     action_rows.append(ar)
   cursor+=n;print(json.dumps({"matched_trials":cursor,"total":len(jobs)}),flush=True)
  finalize(rows,state_rows,action_rows,initial_rows);w.close()
if __name__=="__main__":main()
