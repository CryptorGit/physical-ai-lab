"""Collect bounded student-visited DAgger windows with fail-closed S/W labels."""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import gymnasium as gym
import torch
HERE=Path(__file__).resolve();EXP=HERE.parent.parent;REPO=EXP.parents[2];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion";RAW=OUT/"dagger_dataset"
STOP=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt";WALK=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
sys.path[:0]=[str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),str(REPO/"experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src"),str(EXP/"src")]
import isaaclab_tasks  # noqa:F401,E402
import g1_omnidirectional.tasks  # noqa:F401,E402
from g1_omnidirectional.policy import FrozenGaitActor  # noqa:E402
from g1_omnidirectional.yaw_calibration import calibrate_yaw  # noqa:E402
from g1_explicit_motion_mode.contract import ExplicitMotionModeCommand,MotionMode,build_observation_141,minimum_jerk  # noqa:E402
from g1_explicit_motion_mode.student import ExplicitModeStudent  # noqa:E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa:E402
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli  # noqa:E402
DT=.02
def conditions():
 r=[(22.5*i,.3,0.) for i in range(16)]+[(0.,0.,y) for y in (-.3,.3)]+[(45.*i,.3,y) for i in range(8) for y in (-.3,.3)];return r
def main():
 p=argparse.ArgumentParser();p.add_argument("--round",type=int,required=True);p.add_argument("--horizon",type=int,choices=(8,16,32),required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--episodes-per-condition",type=int,default=20);add_launcher_args(p);a,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0],*hydra];conds=conditions();n=len(conds)*a.episodes_per_condition
 cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=n;cfg.episode_length_s=16.;cfg.seed=20261000+a.round;cfg.observations.policy.enable_corruption=False
 if a.device:cfg.sim.device=agent.device=a.device
 RAW.mkdir(parents=True,exist_ok=True)
 with launch_simulation(cfg,a):
  w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);env=w.unwrapped;dev=env.device;robot=env.scene["robot"];sensor=env.scene["contact_forces"];feet=sensor.find_bodies(".*_ankle_roll_link")[0];term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True;ids=torch.arange(n,device=dev);env.reset(env_ids=ids);obs=w.get_observations().to(dev);gait=torch.zeros(n,device=dev);stop=FrozenGaitActor(STOP).to(dev).eval();walk=FrozenGaitActor(WALK).to(dev).eval();payload=torch.load(a.checkpoint,map_location=dev,weights_only=False);student=ExplicitModeStudent(tuple(payload["architecture"][1:-1])).to(dev);student.load_state_dict(payload["actor_state_dict"]);student.eval();state=ExplicitMotionModeCommand.zeros(n,device=dev);cidx=torch.arange(n,device=dev)//a.episodes_per_condition;eidx=torch.arange(n,device=dev)%a.episodes_per_condition;split=torch.where(eidx<14,0,torch.where(eidx<17,1,2));recipe=(a.round*10**6+torch.arange(n,device=dev)).long();target=torch.zeros(n,3,device=dev)
  for ci,(d,s,y) in enumerate(conds):mask=cidx==ci;rad=math.radians(d);target[mask,0]=s*math.cos(rad);target[mask,1]=s*math.sin(rad);target[mask,2]=y
  fields={k:[] for k in ("observation_141","teacher_action","context","group","condition_index","recipe_id","control_step","split_id","teacher_source","dagger_round","window","labelable")};unlabelable=0
  def command(physical):actor=physical.clone();actor[:,2]=calibrate_yaw(actor[:,2]);term.external_override[:,:3]=actor;term._update_command()
  def safe_mask(done=None):
   contact=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1)>5;upright=robot.data.root_pos_w[:,2]>.5
   return contact.any(1)&upright
  def student_window(mode,teacher,context,group,window,physical_fn):
   nonlocal obs,unlabelable
   active=torch.ones(n,dtype=torch.bool,device=dev)
   for step in range(a.horizon):
    physical,progress=physical_fn(step);state.advance(physical,progress,0. if step==0 else DT);command(physical);obs=w.get_observations().to(dev);x=build_observation_141(obs["policy"],state)
    with torch.inference_mode():label=teacher(obs["policy"],gait);action=student(x)
    labelable=active&safe_mask();unlabelable+=int((active&~labelable).sum());sel=torch.nonzero(labelable).flatten();vals={"observation_141":x,"teacher_action":label,"context":torch.full((n,1),context,device=dev),"group":torch.full((n,1),group,device=dev),"condition_index":cidx[:,None],"recipe_id":recipe[:,None],"control_step":torch.full((n,1),step,device=dev),"split_id":split[:,None],"teacher_source":torch.full((n,1),mode,device=dev),"dagger_round":torch.full((n,1),a.round,device=dev),"window":torch.full((n,1),window,device=dev),"labelable":labelable[:,None]}
    for k,v in vals.items():fields[k].append(v[sel].detach().cpu())
    obs,_,done,extras=w.step(action);obs=obs.to(dev);timeout=extras.get("time_outs",torch.zeros_like(done)).bool();active&=~(done.bool()&~timeout)
  zero=lambda step:(torch.zeros(n,3,device=dev),torch.ones(n,device=dev))
  student_window(0,stop,0,0,0,zero)
  # Teacher roll-in only establishes the next registered boundary; it is not saved.
  for step in range(a.horizon,150):
   state.advance(torch.zeros(n,3,device=dev),torch.ones(n,device=dev),DT);command(state.physical_command);obs=w.get_observations().to(dev)
   with torch.inference_mode():act=stop(obs["policy"],gait)
   obs,_,_,_=w.step(act);obs=obs.to(dev)
  state.request(torch.full((n,),int(MotionMode.WALK),device=dev));start=lambda step:(target*minimum_jerk(torch.full((n,),step/75.,device=dev))[:,None],torch.full((n,),min(1.,step/75.),device=dev))
  student_window(1,walk,1,1,1,start)
  for step in range(a.horizon,226):
   progress=torch.full((n,),min(1.,step/75.),device=dev);physical=target*minimum_jerk(progress)[:,None];state.advance(physical,progress,DT);command(physical);obs=w.get_observations().to(dev)
   with torch.inference_mode():act=walk(obs["policy"],gait)
   obs,_,_,_=w.step(act);obs=obs.to(dev)
  state.request(torch.full((n,),int(MotionMode.STAND),device=dev));stopfn=lambda step:(target*(1.-minimum_jerk(torch.full((n,),step/75.,device=dev)))[:,None],torch.full((n,),min(1.,step/75.),device=dev))
  student_window(0,stop,7,4,2,stopfn)
  data={k:torch.cat(v) for k,v in fields.items()};data["dataset_name"]="Exp014StandOmniWalkTrajectoryDatasetV1_DAgger";data["parent_checkpoint"]=a.checkpoint;data["horizon"]=a.horizon;path=RAW/f"round_{a.round}.pt";torch.save(data,path);print(json.dumps({"path":str(path),"samples":len(data["observation_141"]),"unlabelable_states":unlabelable,"window_counts":{str(i):int((data["window"]==i).sum()) for i in range(3)}},indent=2));w.close()
if __name__=="__main__":main()
