"""Diagnostic-only matched-state one-step branch at the exact-zero start boundary."""
from __future__ import annotations
import argparse,csv,json,math,sys
from collections import defaultdict
from pathlib import Path
import gymnasium as gym
import torch

HERE=Path(__file__).resolve();REPO=HERE.parents[4];EXP=HERE.parent.parent
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
OUT=BASE/"phase_w2_p1_d4_heldout_exact_zero_generalization_diagnosis"
SELECTED=BASE/"phase_w2_p1_r2_long_horizon_group_balanced_stop_integration/raw/selected_student.pt"
PARENT=BASE/"phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
TEACHER=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
sys.path.insert(0,str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"));sys.path.insert(0,str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"));sys.path.insert(0,str(EXP/"src"));sys.path.insert(0,str(HERE.parent))
import isaaclab_tasks  # noqa:F401
import g1_omnidirectional.tasks  # noqa:F401
from g1_omnidirectional.policy import FrozenGaitActor
from g1_omnidirectional.yaw_calibration import calibrate_yaw
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
from train_w2_p1_student import Student

parser=argparse.ArgumentParser();parser.add_argument("--max-envs",type=int,default=1600);add_launcher_args(parser);args,hydra=setup_preset_cli(parser);sys.argv=[sys.argv[0],*hydra]
BRANCHES=("A_STUDENT","A_W1B_LABEL","A_STOP_TEACHER","A_PARENT")

def minjerk(x): x=x.clamp(0,1);return x**3*(10-15*x+6*x*x)
def write_csv(path,rows):
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def finalize(rows,divergence):
 write_csv(OUT/"exact_zero_one_step_counterfactual.csv",rows)
 summary=[]
 for branch in BRANCHES:
  s=[r for r in rows if r["branch"]==branch];times=[r["acquisition_time_s"] for r in s if r["acquisition_time_s"] is not None]
  summary.append({"branch":branch,"trials":len(s),"conditions":24,"moving_endpoint_success":sum(r["moving_endpoint_success"] for r in s)/len(s),"acquisition_success":sum(r["acquisition_success"] for r in s)/len(s),"acquisition_time_median_s":float(torch.tensor(times).median()) if times else None,"fall_rate":sum(r["fall"] for r in s)/len(s),"dangerous_slip_rate":sum(r["dangerous_slip"] for r in s)/len(s),"impact_rate":sum(r["impact"] for r in s)/len(s)})
 div=[]
 for branch in BRANCHES:
  for h in (1,2,4,8,16):
   vals=[r["state_l2_from_student"] for r in divergence if r["branch"]==branch and r["steps_after_branch"]==h];div.append({"branch":branch,"steps_after_branch":h,"mean_state_l2_from_student":sum(vals)/len(vals),"maximum":max(vals)})
 (OUT/"exact_zero_one_step_counterfactual.json").write_text(json.dumps({"diagnostic_only":True,"formal_closed_loop_evaluation":False,"matched_seed_branch_trials_per_condition":200,"matched_physical_state_clone":True,"one_control_step_branch_only":True,"all_subsequent_actions":"step37000 student","summary":summary,"state_divergence":div},indent=2,sort_keys=True)+"\n",encoding="utf-8")

def clone_matched(env,robot,term,episodes):
 origins=env.scene.env_origins;refs=torch.arange(episodes,device=env.device);targets=torch.arange(4*episodes,device=env.device);sources=targets.remainder(episodes)
 local=robot.data.root_pos_w[refs]-origins[refs];root_pose=torch.cat((local[sources]+origins[targets],robot.data.root_quat_w[refs][sources]),1)
 root_vel=torch.cat((robot.data.root_lin_vel_w[refs][sources],robot.data.root_ang_vel_w[refs][sources]),1)
 robot.write_root_pose_to_sim(root_pose,targets);robot.write_root_velocity_to_sim(root_vel,targets);robot.write_joint_state_to_sim(robot.data.joint_pos[refs][sources],robot.data.joint_vel[refs][sources],env_ids=targets)
 for name,value in vars(term).items():
  if isinstance(value,torch.Tensor) and value.ndim and value.shape[0]==env.num_envs:value[targets]=value[refs][sources].clone()
 env.action_manager._action[targets]=env.action_manager.action[refs][sources];env.action_manager._prev_action[targets]=env.action_manager.prev_action[refs][sources];env.episode_length_buf[targets]=env.episode_length_buf[refs][sources];env.sim.forward()

def main():
 jobs=[(d,y,e) for d in range(0,360,45) for y in (-.3,0.,.3) for e in range(200)]
 cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=min(args.max_envs,4*len(jobs));cfg.episode_length_s=12.;cfg.seed=20279001;cfg.observations.policy.enable_corruption=False
 if args.device:cfg.sim.device=agent.device=args.device
 OUT.mkdir(parents=True,exist_ok=True);rows=[];divergence=[]
 with launch_simulation(cfg,args):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);env=wrapped.unwrapped;device=env.device;robot=env.scene["robot"];sensor=env.scene["contact_forces"];feet=sensor.find_bodies(".*_ankle_roll_link")[0];rfeet=robot.find_bodies(".*_ankle_roll_link")[0]
  parent=FrozenGaitActor(PARENT).to(device).eval();teacher=FrozenGaitActor(TEACHER).to(device).eval();state=torch.load(SELECTED,map_location="cpu",weights_only=False)["actor_state_dict"];student=Student(state).to(device).eval();term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True
  max_trials=env.num_envs//4;cursor=0
  while cursor<len(jobs):
   base=jobs[cursor:cursor+max_trials];n=len(base);padded=base+[base[i%n] for i in range(max_trials-n)] if n<max_trials else base;env.reset(env_ids=torch.arange(env.num_envs,device=device));term.external_override.zero_();term._update_command();obs=wrapped.get_observations().to(device);gait=torch.zeros(env.num_envs,device=device)
   # Establish formal stopped states with the exp_012 teacher.
   for step in range(int(round(3/env.step_dt))):
    with torch.inference_mode():a=teacher(obs["policy"],gait)
    obs,_,_,_=wrapped.step(a);obs=obs.to(device)
   clone_matched(env,robot,term,max_trials);term.external_override.zero_();term._update_command();obs=wrapped.get_observations().to(device)
   with torch.inference_mode():sa=student(obs["policy"],gait);pa=parent(obs["policy"],gait);ta=teacher(obs["policy"],gait)
   actions=torch.empty_like(sa);actions[:max_trials]=sa[:max_trials];actions[max_trials:2*max_trials]=pa[max_trials:2*max_trials];actions[2*max_trials:3*max_trials]=ta[2*max_trials:3*max_trials];actions[3*max_trials:]=pa[3*max_trials:]
   # Exact-zero boundary branch: this one action only.
   obs,_,done,extras=wrapped.step(actions);obs=obs.to(device)
   fall=done.bool()&~extras.get("time_outs",torch.zeros_like(done)).bool();slip=torch.zeros_like(fall);impact=torch.zeros_like(fall);streak=torch.zeros(env.num_envs,dtype=torch.long,device=device);speed_sum=torch.zeros(env.num_envs,device=device);yaw_sum=torch.zeros_like(speed_sum);vec_err=torch.zeros_like(speed_sum);yaw_err=torch.zeros_like(speed_sum);measure=torch.zeros_like(speed_sum);first_move=torch.full_like(speed_sum,float("nan"));acq=torch.full_like(speed_sum,float("nan"));sustain=torch.zeros(env.num_envs,dtype=torch.long,device=device)
   directions=torch.tensor([math.radians(x[0]) for x in padded],device=device);yaw_base=torch.tensor([x[1] for x in padded],device=device);target_base=torch.stack((.3*directions.cos(),.3*directions.sin(),yaw_base),1);target=target_base.repeat(4,1)
   horizons={1,2,4,8,16}
   total=int(round((1.5+4.0)/env.step_dt))
   for k in range(1,total+1):
    t=k*env.step_dt;alpha=minjerk(torch.tensor(t/1.5,device=device));physical=target*alpha;term.external_override[:,:2]=physical[:,:2];term.external_override[:,2]=calibrate_yaw(physical[:,2]);
    if k==1:term._update_command();obs=wrapped.get_observations().to(device)
    with torch.inference_mode():a=student(obs["policy"],gait)
    obs,_,dn,ex=wrapped.step(a);obs=obs.to(device);fall|=dn.bool()&~ex.get("time_outs",torch.zeros_like(dn)).bool();force=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);fs=torch.linalg.vector_norm(robot.data.body_lin_vel_w[:,rfeet,:2],dim=-1);bad=((fs>.55)&(force>5)).any(-1);streak=torch.where(bad,streak+1,torch.zeros_like(streak));slip|=streak>=5;impact|=force.amax(-1)>3500
    actual=robot.data.root_lin_vel_b[:,:2];ay=robot.data.root_ang_vel_b[:,2];spd=torch.linalg.vector_norm(actual,dim=-1);new=torch.isnan(first_move)&(spd>.08);first_move[new]=t
    ve=torch.linalg.vector_norm(actual-target[:,:2],dim=-1);ye=(ay-target[:,2]).abs();sign=(target[:,2].abs()<1e-8)|((ay*target[:,2])>0);endpoint=(ve<=.25)&(ye<=.20)&sign&~fall&~slip&~impact
    if t>=1.5:
     sustain=torch.where(endpoint,sustain+1,torch.zeros_like(sustain));newa=torch.isnan(acq)&(sustain>=int(round(.2/env.step_dt)));acq[newa]=torch.tensor(t-1.5,device=device)
    else:sustain.zero_()
    if t>=3.5:speed_sum+=spd;yaw_sum+=ay.abs();vec_err+=ve;yaw_err+=ye;measure+=1
    if k in horizons:
     state_now=torch.cat((robot.data.root_lin_vel_b,robot.data.root_ang_vel_b,robot.data.joint_pos,robot.data.joint_vel),1);ref=state_now[:max_trials]
     for bi,name in enumerate(BRANCHES):
      cur=state_now[bi*max_trials:(bi+1)*max_trials];dist=torch.linalg.vector_norm(cur-ref,dim=1)
      for ci,(d,y,e) in enumerate(padded[:n]):divergence.append({"direction":d,"yaw":y,"episode":e,"branch":name,"steps_after_branch":k,"state_l2_from_student":float(dist[ci])})
   endpoint=(vec_err/measure<=.25)&(yaw_err/measure<=.20)&~fall&~slip&~impact
   for bi,name in enumerate(BRANCHES):
    for i,(d,y,e) in enumerate(padded[:n]):
     j=bi*max_trials+i;rows.append({"direction":d,"yaw":y,"episode":e,"branch":name,"matched_state_group":cursor+i,"moving_endpoint_success":int(endpoint[j]),"acquisition_success":int(torch.isfinite(acq[j]) and acq[j]<=3),"acquisition_time_s":None if torch.isnan(acq[j]) else float(acq[j]),"first_movement_time_s":None if torch.isnan(first_move[j]) else float(first_move[j]),"translation_mae":float(vec_err[j]/measure[j]),"yaw_mae":float(yaw_err[j]/measure[j]),"fall":int(fall[j]),"dangerous_slip":int(slip[j]),"impact":int(impact[j])})
   cursor+=n;print(json.dumps({"matched_trials":cursor,"total":len(jobs)}),flush=True)
  finalize(rows,divergence);wrapped.close()

if __name__=="__main__":main()
