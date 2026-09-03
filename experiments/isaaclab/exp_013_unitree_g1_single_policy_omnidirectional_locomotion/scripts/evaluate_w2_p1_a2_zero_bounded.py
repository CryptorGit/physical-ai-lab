"""Safety-capped two-second zero-command diagnostic for the frozen R2 student."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import gymnasium as gym
import torch
HERE=Path(__file__).resolve();REPO=HERE.parents[4];EXP=HERE.parent.parent;BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion";OUT=BASE/"phase_w2_p1_a2_start_boundary_physical_diagnosis";SELECTED=BASE/"phase_w2_p1_r2_long_horizon_group_balanced_stop_integration/raw/selected_student.pt";TEACHER=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
sys.path[:0]=[str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),str(EXP/"src"),str(HERE.parent)]
import isaaclab_tasks  # noqa:F401
import g1_omnidirectional.tasks  # noqa:F401
from g1_omnidirectional.policy import FrozenGaitActor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
from train_w2_p1_student import Student
parser=argparse.ArgumentParser();parser.add_argument("--episodes-per-group",type=int,default=400);add_launcher_args(parser);args,hydra=setup_preset_cli(parser);sys.argv=[sys.argv[0],*hydra]
def main():
 groups=("STANDARD_RESET","EXP012_STEADY_STOP","START_RETENTION_EXACT_ZERO");n=args.episodes_per_group;cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=n*len(groups);cfg.episode_length_s=8.;cfg.seed=20279021;cfg.observations.policy.enable_corruption=False
 if args.device:cfg.sim.device=agent.device=args.device
 rows=[]
 with launch_simulation(cfg,args):
  w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);env=w.unwrapped;dev=env.device;robot=env.scene["robot"];sensor=env.scene["contact_forces"];feet=sensor.find_bodies(".*_ankle_roll_link")[0];rfeet=robot.find_bodies(".*_ankle_roll_link")[0];teacher=FrozenGaitActor(TEACHER).to(dev).eval();student=Student(torch.load(SELECTED,map_location="cpu",weights_only=False)["actor_state_dict"]).to(dev).eval();term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True;term.external_override.zero_();term._update_command();obs=w.get_observations().to(dev);gait=torch.zeros(env.num_envs,device=dev)
  settle=torch.arange(n,3*n,device=dev)
  for _ in range(round(3/env.step_dt)):
   with torch.inference_mode():a=student(obs["policy"],gait);a[settle]=teacher(obs["policy"][settle],gait[settle])
   obs,_,_,_=w.step(a);obs=obs.to(dev)
  fall=torch.zeros(env.num_envs,dtype=torch.bool,device=dev);slip=torch.zeros_like(fall);impact=torch.zeros_like(fall);tilt=torch.zeros_like(fall);streak=torch.zeros(env.num_envs,dtype=torch.long,device=dev);speed=torch.zeros(env.num_envs,device=dev);yaw=torch.zeros_like(speed);foot=torch.zeros_like(speed);contact_switch=torch.zeros_like(speed);last_contact=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1)>5;steps=0
  for _ in range(round(2/env.step_dt)):
   with torch.inference_mode():a=student(obs["policy"],gait)
   obs,_,dn,ex=w.step(a);obs=obs.to(dev);fall|=dn.bool()&~ex.get("time_outs",torch.zeros_like(dn)).bool();force=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);contact=force>5;contact_switch+=(contact!=last_contact).any(1);last_contact=contact;fs=torch.linalg.vector_norm(robot.data.body_lin_vel_w[:,rfeet,:2],dim=-1);bad=(fs>.55)&contact;streak=torch.where(bad.any(1),streak+1,torch.zeros_like(streak));slip|=streak>=5;impact|=force.amax(1)>3500;g=robot.data.projected_gravity_b;tilt|=(torch.atan2(g[:,:2].norm(dim=1),g[:,2].abs().clamp_min(1e-6))>.7);speed+=torch.linalg.vector_norm(robot.data.root_lin_vel_b[:,:2],dim=1);yaw+=robot.data.root_ang_vel_b[:,2].abs();foot+=fs.mean(1);steps+=1
  for gi,name in enumerate(groups):
   for i in range(n):
    j=gi*n+i;rows.append({"initial_state_group":name,"episode":i,"survived":int(not (fall[j] or tilt[j] or slip[j] or impact[j])),"fall":int(fall[j]),"dangerous_slip":int(slip[j]),"impact":int(impact[j]),"excessive_tilt":int(tilt[j]),"mean_speed":float(speed[j]/steps),"mean_abs_yaw":float(yaw[j]/steps),"mean_foot_speed":float(foot[j]/steps),"contact_switch_rate":float(contact_switch[j]/2.)})
  OUT.mkdir(parents=True,exist_ok=True);keys=list(rows[0]);
  with (OUT/"student_zero_command_bounded_diagnostic.csv").open("w",newline="",encoding="utf-8") as f:wri=csv.DictWriter(f,fieldnames=keys);wri.writeheader();wri.writerows(rows)
  summary=[]
  for name in groups:
   s=[r for r in rows if r["initial_state_group"]==name];summary.append({"initial_state_group":name,"episodes":len(s),**{k:sum(r[k] for r in s)/len(s) for k in ("survived","fall","dangerous_slip","impact","excessive_tilt","mean_speed","mean_abs_yaw","mean_foot_speed","contact_switch_rate")}})
  (OUT/"student_zero_command_bounded_diagnostic.json").write_text(json.dumps({"diagnostic_only":True,"duration_s":2,"hard_termination_contract":["fall","tilt","dangerous_slip","impact","nonfinite"],"summary":summary},indent=2)+"\n");w.close()
if __name__=="__main__":main()
