"""Formal single-actor STAND -> WALK/yaw -> STAND sequence evaluation."""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import gymnasium as gym
import torch

HERE=Path(__file__).resolve();EXP=HERE.parent.parent;REPO=EXP.parents[2];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion";RAW=OUT/"phase2_batches"
sys.path[:0]=[str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),str(REPO/"experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src"),str(EXP/"src")]
import isaaclab_tasks  # noqa:F401,E402
import g1_omnidirectional.tasks  # noqa:F401,E402
from g1_omnidirectional.yaw_calibration import calibrate_yaw  # noqa:E402
from g1_explicit_motion_mode.contract import ExplicitMotionModeCommand,MotionMode,build_observation_141,minimum_jerk  # noqa:E402
from g1_explicit_motion_mode.student import ExplicitModeStudent  # noqa:E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa:E402
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli  # noqa:E402

DT=.02;RAMP=75
def all_conditions():
 rows=[{"id":f"walk_{i:02d}","kind":"walk","direction":22.5*i,"speed":.3,"yaw":0.} for i in range(16)]
 rows += [{"id":f"pure_yaw_{'neg' if y<0 else 'pos'}","kind":"pure_yaw","direction":0.,"speed":0.,"yaw":y} for y in (-.3,.3)]
 rows += [{"id":f"moving_yaw_{i:02d}_{'neg' if y<0 else 'pos'}","kind":"moving_yaw","direction":45.*i,"speed":.3,"yaw":y} for i in range(8) for y in (-.3,.3)]
 return rows
def sustained(trace,width):
 streak=torch.zeros(trace.shape[1],dtype=torch.long,device=trace.device);passed=torch.zeros(trace.shape[1],dtype=torch.bool,device=trace.device)
 for v in trace:streak=torch.where(v,streak+1,torch.zeros_like(streak));passed|=streak>=width
 return passed
def main():
 p=argparse.ArgumentParser();p.add_argument("--batch",type=int,required=True);p.add_argument("--conditions-per-batch",type=int,default=8);p.add_argument("--checkpoint",required=True);p.add_argument("--output-tag",default="phase2_batches");p.add_argument("--episodes",type=int,default=100);p.add_argument("--seed-base",type=int,default=20260900);add_launcher_args(p);a,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0],*hydra]
 specs=all_conditions()[a.batch*a.conditions_per_batch:(a.batch+1)*a.conditions_per_batch]
 if not specs:return
 n=len(specs)*a.episodes;cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=n;cfg.episode_length_s=18.;cfg.seed=a.seed_base+a.batch;cfg.observations.policy.enable_corruption=False
 if a.device:cfg.sim.device=agent.device=a.device
 raw=OUT/a.output_tag;raw.mkdir(parents=True,exist_ok=True)
 with launch_simulation(cfg,a):
  w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);env=w.unwrapped;dev=env.device;robot=env.scene["robot"];sensor=env.scene["contact_forces"];sf=sensor.find_bodies(".*_ankle_roll_link")[0];rf=robot.find_bodies(".*_ankle_roll_link")[0];term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True;ids=torch.arange(n,device=dev);env.reset(env_ids=ids);obs=w.get_observations().to(dev)
  payload=torch.load(a.checkpoint,map_location=dev,weights_only=False);model=ExplicitModeStudent(tuple(payload["architecture"][1:-1])).to(dev);model.load_state_dict(payload["actor_state_dict"]);model.eval();state=ExplicitMotionModeCommand.zeros(n,device=dev);condition=torch.arange(n,device=dev)//a.episodes;target=torch.zeros(n,3,device=dev)
  for ci,s in enumerate(specs):
   mask=condition==ci;rad=math.radians(s["direction"]);target[mask,0]=s["speed"]*math.cos(rad);target[mask,1]=s["speed"]*math.sin(rad);target[mask,2]=s["yaw"]
  limits=robot.data.joint_vel_limits;limits=limits[...,1].abs() if limits.ndim==3 else limits
  fall=torch.zeros(n,dtype=torch.bool,device=dev);slip=fall.clone();impact=fall.clone();sat=fall.clone();ss=torch.zeros(n,dtype=torch.long,device=dev);sats=ss.clone()
  def safety(done,extras):
   nonlocal fall,slip,impact,sat,ss,sats
   timeout=extras.get("time_outs",torch.zeros_like(done)).bool();fall|=done.bool()&~timeout;force=sensor.data.net_forces_w_history[:,-1,sf,:].norm(dim=-1);contact=force>5;feet_speed=torch.linalg.vector_norm(robot.data.body_lin_vel_w[:,rf,:2],dim=-1);bad=((feet_speed>.55)&contact).any(1);ss=torch.where(bad,ss+1,torch.zeros_like(ss));slip|=ss>=5;impact|=force.amax(1)>3500;ratio=robot.data.joint_vel.abs().div(limits.clamp_min(1e-6)).amax(1);sats=torch.where(ratio>.95,sats+1,torch.zeros_like(sats));sat|=sats>=5;return contact
  def set_command(physical):
   actor=physical.clone();actor[:,2]=calibrate_yaw(actor[:,2]);term.external_override[:,:3]=actor;term._update_command()
  def act():
   with torch.inference_mode():return model(build_observation_141(obs["policy"],state))
  # Formal 2-second practical STAND from reset.
  stand_speed=[];stand_yaw=[]
  for step in range(100):
   physical=torch.zeros(n,3,device=dev);state.advance(physical,torch.ones(n,device=dev),DT);set_command(physical);obs=w.get_observations().to(dev);action=act();obs,_,done,extras=w.step(action);obs=obs.to(dev);safety(done,extras);stand_speed.append(torch.linalg.vector_norm(robot.data.root_lin_vel_b[:,:2],dim=1));stand_yaw.append(robot.data.root_ang_vel_b[:,2].abs())
  stand_mean_speed=torch.stack(stand_speed).mean(0);stand_mean_yaw=torch.stack(stand_yaw).mean(0);stand_ok=(stand_mean_speed<=.08)&(stand_mean_yaw<=.08)&~fall&~slip&~impact
  # Immediate WALK mode request, then ramp and hold.
  state.request(torch.full((n,),int(MotionMode.WALK),device=dev));trace=[];end_v=[];end_y=[]
  for step in range(RAMP+200):
   progress=torch.tensor(min(1.,step/RAMP),device=dev).expand(n);physical=target*minimum_jerk(progress)[:,None];state.advance(physical,progress,0. if step==0 else DT);set_command(physical);obs=w.get_observations().to(dev);action=act();obs,_,done,extras=w.step(action);obs=obs.to(dev);contact=safety(done,extras);actual=robot.data.root_lin_vel_b[:,:2];yaw=robot.data.root_ang_vel_b[:,2];ve=torch.linalg.vector_norm(actual-physical[:,:2],dim=1);asp=torch.linalg.vector_norm(actual,dim=1);tsp=torch.linalg.vector_norm(physical[:,:2],dim=1);ta=torch.atan2(physical[:,1],physical[:,0]);aa=torch.atan2(actual[:,1],actual[:,0]);de=torch.atan2(torch.sin(aa-ta),torch.cos(aa-ta)).abs()*180/math.pi;translation=torch.where(tsp<1e-8,asp<=.08,ve<=.25);direction=torch.where(tsp<1e-8,torch.ones_like(translation),de<=25);yawok=torch.where(physical[:,2].abs()<1e-8,yaw.abs()<=.2,(torch.sign(yaw)==torch.sign(physical[:,2]))&((yaw-physical[:,2]).abs()<=.2));combined=translation&direction&yawok&contact.any(1)&~fall&~slip&~impact
   if step>=RAMP:trace.append(combined)
   if step>=175:end_v.append(actual);end_y.append(yaw)
  trace=torch.stack(trace);acq=sustained(trace[:150],10);mv=torch.stack(end_v).mean(0);my=torch.stack(end_y).mean(0);vector_error=torch.linalg.vector_norm(mv-target[:,:2],dim=1);target_angle=torch.atan2(target[:,1],target[:,0]);mean_angle=torch.atan2(mv[:,1],mv[:,0]);direction_error=torch.atan2(torch.sin(mean_angle-target_angle),torch.cos(mean_angle-target_angle)).abs()*180/math.pi;speed=torch.linalg.vector_norm(target[:,:2],dim=1);translation_end=torch.where(speed<1e-8,torch.linalg.vector_norm(mv,dim=1)<=.08,(vector_error<=.25)&(direction_error<=25));yaw_error=(my-target[:,2]).abs();yaw_end=torch.where(target[:,2].abs()<1e-8,my.abs()<=.2,(torch.sign(my)==torch.sign(target[:,2]))&(yaw_error<=.2));endpoint=translation_end&yaw_end&~fall&~slip&~impact&~sat
  # STOP mode is visible before minimum-jerk deceleration begins.
  state.request(torch.full((n,),int(MotionMode.STAND),device=dev));stop_speed=[];stop_yaw=[]
  for step in range(RAMP+100):
   progress=torch.tensor(min(1.,step/RAMP),device=dev).expand(n);physical=target*(1.-minimum_jerk(progress))[:,None];state.advance(physical,progress,0. if step==0 else DT);set_command(physical);obs=w.get_observations().to(dev);action=act();obs,_,done,extras=w.step(action);obs=obs.to(dev);safety(done,extras)
   if step>=RAMP:stop_speed.append(torch.linalg.vector_norm(robot.data.root_lin_vel_b[:,:2],dim=1));stop_yaw.append(robot.data.root_ang_vel_b[:,2].abs())
  final_speed=torch.stack(stop_speed).mean(0);final_yaw=torch.stack(stop_yaw).mean(0);stop_ok=(final_speed<=.08)&(final_yaw<=.08)&~fall&~slip&~impact
  rows=[]
  for ci,s in enumerate(specs):
   mask=condition==ci;rows.append({**s,"episodes":a.episodes,"stand_hold":float(stand_ok[mask].float().mean()),"stand_mean_speed":float(stand_mean_speed[mask].mean()),"stand_mean_abs_yaw":float(stand_mean_yaw[mask].mean()),"endpoint":float(endpoint[mask].float().mean()),"acquisition_0p20":float(acq[mask].float().mean()),"walk_to_stand":float(stop_ok[mask].float().mean()),"final_speed":float(final_speed[mask].mean()),"final_abs_yaw":float(final_yaw[mask].mean()),"full_sequence":float((stand_ok&endpoint&stop_ok)[mask].float().mean()),"fall_rate":float(fall[mask].float().mean()),"dangerous_slip_rate":float(slip[mask].float().mean()),"impact_rate":float(impact[mask].float().mean()),"saturation_rate":float(sat[mask].float().mean())})
  path=raw/f"batch_{a.batch:02d}.json";path.write_text(json.dumps({"checkpoint":str(Path(a.checkpoint).resolve()),"runtime":{"actors":1,"teachers":0,"routers":0,"checkpoint_switching":0,"action_blending":0},"rows":rows},indent=2)+"\n");print(json.dumps({"path":str(path),"rows":rows},indent=2));w.close()
if __name__=="__main__":main()
