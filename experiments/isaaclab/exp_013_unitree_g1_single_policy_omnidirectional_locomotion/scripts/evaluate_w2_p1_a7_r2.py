"""Fresh-process V2 start evaluation for A7-R2 checkpoints."""
from __future__ import annotations
import argparse,csv,json,math,sys
from collections import defaultdict
from pathlib import Path
import gymnasium as gym
import torch
HERE=Path(__file__).resolve();REPO=HERE.parents[4];EXP=HERE.parent.parent;BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion";M0=BASE/"phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight";TEACHER=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
sys.path[:0]=[str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),str(EXP/"src")]
import isaaclab_tasks,g1_omnidirectional.tasks
from g1_omnidirectional.policy import FrozenGaitActor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
p=argparse.ArgumentParser();p.add_argument("--policy",required=True);p.add_argument("--batch",type=int,required=True);p.add_argument("--split",choices=("validation","heldout"),required=True);p.add_argument("--mode",choices=("guard","timeline","formal_matrix","pure_yaw","rear_boundary"),required=True);p.add_argument("--condition-index",type=int);p.add_argument("--episodes",type=int);p.add_argument("--output",required=True);add_launcher_args(p);a,h=setup_preset_cli(p);sys.argv=[sys.argv[0],*h]
N=1024;ROLL=150
def mj(x):x=x.clamp(0.,1.);return 10*x**3-15*x**4+6*x**5
def conditions(mode):
 if mode=="guard":
  z=[{"group":"zero_yaw","direction":i*22.5,"speed":.3,"yaw":0.} for i in range(16)];m=[{"group":"moving_turn","direction":i*45.,"speed":.3,"yaw":y} for i in range(8) for y in (-.3,0.,.3)];return z+m+[{"group":"forward_anchor","direction":0.,"speed":s,"yaw":0.} for s in (.6,1.2)]+[{"group":"rear_0p15","direction":180.,"speed":.15,"yaw":y} for y in (-.3,.3)]+[{"group":"pure_yaw","direction":0.,"speed":0.,"yaw":y} for y in (-.3,.3)]
 if mode=="timeline":return [{"group":"rear","direction":180.,"speed":s,"yaw":y} for s in (.15,.20,.25,.30) for y in (-.3,.3)]
 if mode=="formal_matrix":return [{"group":"start_matrix","direction":i*45.,"speed":.3,"yaw":y} for i in range(8) for y in (-.3,0.,.3)]
 if mode=="pure_yaw":return [{"group":"pure_yaw","direction":0.,"speed":0.,"yaw":y} for y in (-.3,.3)]
 return [{"group":"rear_boundary","direction":180.,"speed":s,"yaw":y} for s in (.10,.15,.20,.25,.30,.35) for y in (-.3,.3)]
cfg,ac=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=N;cfg.episode_length_s=12.;cfg.seed=20278501;cfg.observations.policy.enable_corruption=False
if a.device:cfg.sim.device=ac.device=a.device
masks=json.loads((M0/"a7_environment_masks.json").read_text())["batches"][str(a.batch)];active_cpu=torch.tensor(masks[f"{a.split}_mask"],dtype=torch.bool);conds=conditions(a.mode)
if a.condition_index is not None:conds=[conds[a.condition_index]]
if a.episodes is not None:
 active_ids_cpu=torch.nonzero(active_cpu).flatten();active_cpu.zero_();active_cpu[active_ids_cpu[:a.episodes]]=True
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
with launch_simulation(cfg,a):
 w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=ac.clip_actions);env=w.unwrapped;robot=env.scene["robot"];sensor=env.scene["contact_forces"];feet=sensor.find_bodies(".*_ankle_roll_link")[0];rfeet=robot.find_bodies(".*_ankle_roll_link")[0];term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True;teacher=FrozenGaitActor(TEACHER).to(env.device).eval();g=torch.zeros(N,device=env.device);ids=torch.arange(N,device=env.device);limits=robot.data.joint_vel_limits;limits=limits[...,1].abs() if limits.ndim==3 else limits
 for batch_id in range(a.batch+1):
  env.reset(env_ids=ids);term.external_override.zero_();term._update_command();obs=w.get_observations().to(env.device)
  for _ in range(ROLL):
   with torch.inference_mode():act=teacher(obs["policy"],g)
   obs,_,_,_=w.step(act);obs=obs.to(env.device)
 policy=FrozenGaitActor(Path(a.policy)).to(env.device).eval();active=active_cpu.to(env.device);active_ids=torch.nonzero(active).flatten();assign=torch.full((N,),-1,dtype=torch.long,device=env.device)
 for j,e in enumerate(active_ids.tolist()):assign[e]=j%len(conds)
 target=torch.zeros(N,3,device=env.device)
 for ci,c in enumerate(conds):
  q=assign==ci;r=math.radians(c["direction"]);target[q,0]=c["speed"]*math.cos(r);target[q,1]=c["speed"]*math.sin(r);target[q,2]=c["yaw"]
 fall=torch.zeros(N,dtype=torch.bool,device=env.device);slip=fall.clone();impact=fall.clone();sat=fall.clone();slipst=torch.zeros(N,dtype=torch.long,device=env.device);satst=slipst.clone();streak=torch.zeros(N,dtype=torch.long,device=env.device);acquired=torch.zeros(N,dtype=torch.bool,device=env.device);acq_step=torch.full((N,),-1,dtype=torch.long,device=env.device);resets=torch.zeros(N,dtype=torch.long,device=env.device);longest=torch.zeros(N,dtype=torch.long,device=env.device);endpoint_n=torch.zeros(N,dtype=torch.long,device=env.device);endpoint_vel=torch.zeros(N,2,device=env.device);endpoint_vec_error=torch.zeros(N,device=env.device);endpoint_yaw=torch.zeros(N,device=env.device);endpoint_yaw_error=torch.zeros(N,device=env.device)
 for step in range(275):
  alpha=mj(torch.tensor(step/75.,device=env.device));physical=target*alpha;actor_cmd=physical.clone();actor_cmd[:,2]=torch.where(actor_cmd[:,2]>0,actor_cmd[:,2]*1.5,actor_cmd[:,2]);term.external_override.zero_();term.external_override[active]=actor_cmd[active];term._update_command();obs=w.get_observations().to(env.device)
  with torch.inference_mode():pa=policy(obs["policy"],g);ha=teacher(obs["policy"],g);action=torch.where(active[:,None],pa,ha)
  obs,_,done,extra=w.step(action);obs=obs.to(env.device);timeout=extra.get("time_outs",torch.zeros_like(done)).bool();fall|=done.bool()&~timeout;force=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);contact=force>5;fs=torch.linalg.vector_norm(robot.data.body_lin_vel_w[:,rfeet,:2],dim=-1);bad=((fs>.55)&contact).any(1);slipst=torch.where(bad,slipst+1,torch.zeros_like(slipst));slip|=slipst>=5;impact|=force.amax(1)>3500;ratio=robot.data.joint_vel.abs().div(limits.clamp_min(1e-6)).amax(1);satst=torch.where(ratio>.95,satst+1,torch.zeros_like(satst));sat|=satst>=5
  actual=robot.data.root_lin_vel_b[:,:2];ay=robot.data.root_ang_vel_b[:,2];vec=torch.linalg.vector_norm(actual-target[:,:2],dim=1);speed=torch.linalg.vector_norm(actual,dim=1);ta=torch.atan2(target[:,1],target[:,0]);aa=torch.atan2(actual[:,1],actual[:,0]);de=torch.atan2(torch.sin(aa-ta),torch.cos(aa-ta)).abs()*180/math.pi;trans=torch.where(torch.linalg.vector_norm(target[:,:2],dim=1)<1e-8,speed<=.08,vec<=.25);direction=torch.where(torch.linalg.vector_norm(target[:,:2],dim=1)<1e-8,torch.ones_like(trans),de<=25);yawok=torch.where(target[:,2].abs()<1e-8,ay.abs()<=.2,(torch.sign(ay)==torch.sign(target[:,2]))&((ay-target[:,2]).abs()<=.2));safe=~fall&~slip&~impact;passing=trans&direction&yawok&contact.any(1)&safe
  if step>=75:
   resets+=(~passing)&(streak>0);streak=torch.where(passing,streak+1,torch.zeros_like(streak));longest=torch.maximum(longest,streak);hit=(streak>=10)&~acquired;acquired|=hit;acq_step[hit]=step-9-75
  if step>=175:
   endpoint_n+=active;endpoint_vel+=actual*active[:,None];endpoint_vec_error+=vec*active;endpoint_yaw+=ay*active;endpoint_yaw_error+=(ay-target[:,2]).abs()*active
 endpoint_count=endpoint_n.clamp_min(1);mean_vel=endpoint_vel/endpoint_count[:,None];mean_yaw=endpoint_yaw/endpoint_count;mean_vec_error=endpoint_vec_error/endpoint_count;mean_yaw_error=endpoint_yaw_error/endpoint_count;target_speed=torch.linalg.vector_norm(target[:,:2],dim=1);mean_speed=torch.linalg.vector_norm(mean_vel,dim=1);mean_angle=torch.atan2(mean_vel[:,1],mean_vel[:,0]);target_angle=torch.atan2(target[:,1],target[:,0]);mean_direction_error=torch.atan2(torch.sin(mean_angle-target_angle),torch.cos(mean_angle-target_angle)).abs()*180/math.pi;endpoint_translation=torch.where(target_speed<1e-8,mean_speed<=.08,(mean_vec_error<=.25)&(mean_direction_error<=25));endpoint_yaw_ok=torch.where(target[:,2].abs()<1e-8,mean_yaw.abs()<=.2,(torch.sign(mean_yaw)==torch.sign(target[:,2]))&(mean_yaw_error<=.2));endpoint_ok=endpoint_translation&endpoint_yaw_ok&~fall&~slip&~impact&~sat
 rows=[]
 for ci,c in enumerate(conds):
  q=assign==ci;n=int(q.sum());times=acq_step[q].float()*.02;valid=times[times>=0];rows.append({**c,"episodes":n,"endpoint_success":float(endpoint_ok[q].float().mean()),"acquisition_0p10":float((longest[q]>=5).float().mean()),"acquisition_0p20":float(acquired[q].float().mean()),"acquisition_median_s":float(valid.median()) if len(valid) else None,"acquisition_p95_s":float(torch.quantile(valid,.95)) if len(valid) else None,"yaw_mae":float(mean_yaw_error[q].mean()),"yaw_timer_resets":float(resets[q].float().mean()),"longest_yaw_pass_s":float(longest[q].float().mean()*.02),"fall_rate":float(fall[q].float().mean()),"dangerous_slip_rate":float(slip[q].float().mean()),"impact_rate":float(impact[q].float().mean()),"saturation_rate":float(sat[q].float().mean())})
 with out.open("w",newline="",encoding="utf-8") as f:wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 out.with_suffix(".json").write_text(json.dumps({"mode":a.mode,"batch":a.batch,"split":a.split,"policy":str(a.policy),"rows":rows},indent=2)+"\n",encoding="utf-8");print(json.dumps({"mode":a.mode,"active":int(active.sum()),"rows":len(rows)},indent=2),flush=True);w.close()
