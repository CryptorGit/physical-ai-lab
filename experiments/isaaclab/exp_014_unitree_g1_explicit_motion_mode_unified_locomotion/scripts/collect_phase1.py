"""Collect new, formally scoped S/W trajectory labels for EXP 014 Phase 1."""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import gymnasium as gym
import torch

HERE=Path(__file__).resolve(); EXP=HERE.parent.parent; REPO=EXP.parents[2]
OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion"; RAW=OUT/"phase1_dataset"
STOP=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
WALK=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
sys.path[:0]=[str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),str(REPO/"experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src"),str(EXP/"src")]
import isaaclab_tasks  # noqa:F401,E402
import g1_omnidirectional.tasks  # noqa:F401,E402
from g1_omnidirectional.policy import FrozenGaitActor  # noqa:E402
from g1_omnidirectional.yaw_calibration import calibrate_yaw  # noqa:E402
from g1_explicit_motion_mode.contract import ExplicitMotionModeCommand,MotionMode,build_observation_141,minimum_jerk  # noqa:E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa:E402
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli  # noqa:E402

CONTEXTS=["STAND_HOLD","STAND_TO_WALK_B0","STAND_TO_WALK_RAMP","STAND_TO_WALK_ACQUISITION","WALK_STEADY","WALK_PURE_YAW","WALK_MOVING_YAW","WALK_TO_STAND_DECELERATION","WALK_TO_STAND_RECOVERY","STAND_AFTER_STOP"]
GROUPS=["STAND_HOLD","STAND_TO_WALK","WALK_STEADY","WALK_YAW","WALK_TO_STAND","STAND_AFTER_STOP"]
DT=.02; RAMP=75

def conditions():
 rows=[]
 for d in range(0,360,22):
  # Use exact 22.5-degree grid rather than integer range approximation.
  pass
 rows=[{"kind":"walk","direction":22.5*i,"speed":.3,"yaw":0.} for i in range(16)]
 rows += [{"kind":"pure_yaw","direction":0.,"speed":0.,"yaw":y} for y in (-.3,.3)]
 rows += [{"kind":"moving_yaw","direction":45.*i,"speed":.3,"yaw":y} for i in range(8) for y in (-.3,.3)]
 return rows

def split_for_episode(index):
 r=index%20
 return 0 if r<14 else (1 if r<17 else 2)

def main():
 p=argparse.ArgumentParser();p.add_argument("--batch",type=int,default=0);p.add_argument("--episodes-per-condition",type=int,default=20);add_launcher_args(p);args,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0],*hydra]
 conds=conditions(); n=len(conds)*args.episodes_per_condition
 cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=n;cfg.episode_length_s=16.;cfg.seed=20260803+args.batch;cfg.observations.policy.enable_corruption=False
 if args.device:cfg.sim.device=agent.device=args.device
 RAW.mkdir(parents=True,exist_ok=True)
 with launch_simulation(cfg,args):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);env=wrapped.unwrapped;dev=env.device;term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True
  stop=FrozenGaitActor(STOP).to(dev).eval();walk=FrozenGaitActor(WALK).to(dev).eval();gait=torch.zeros(n,device=dev);ids=torch.arange(n,device=dev);env.reset(env_ids=ids)
  cidx=torch.arange(n,device=dev)//args.episodes_per_condition;eidx=torch.arange(n,device=dev)%args.episodes_per_condition
  target=torch.zeros(n,3,device=dev); kinds=[]
  for ci,c in enumerate(conds):
   mask=cidx==ci;rad=math.radians(c["direction"]);target[mask,0]=c["speed"]*math.cos(rad);target[mask,1]=c["speed"]*math.sin(rad);target[mask,2]=c["yaw"];kinds.append(c["kind"])
  recipe=(args.batch*n+torch.arange(n,device=dev)).long();split=torch.tensor([split_for_episode(int(x)+args.batch*args.episodes_per_condition) for x in eidx],device=dev)
  fields={k:[] for k in ("observation_141","teacher_action","context","group","physical_command","target_mode","previous_target_mode","previous_physical_command","command_delta","time_since_mode_change","ramp_progress","condition_index","recipe_id","episode_id","control_step","split_id","teacher_source")}
  state=ExplicitMotionModeCommand.zeros(n,device=dev);obs=wrapped.get_observations().to(dev);sample_counter=0
  def env_command(physical):
   actor=physical.clone();actor[:,2]=calibrate_yaw(actor[:,2]);term.external_override[:,:3]=actor;term._update_command()
  def record(context,group,action,teacher_source,step):
   nonlocal sample_counter
   x=build_observation_141(obs["policy"],state); count=len(x)
   values={"observation_141":x,"teacher_action":action,"context":torch.full((count,1),CONTEXTS.index(context),device=dev),"group":torch.full((count,1),GROUPS.index(group),device=dev),"physical_command":state.physical_command,"target_mode":state.target_mode[:,None],"previous_target_mode":state.previous_target_mode[:,None],"previous_physical_command":state.previous_physical_command,"command_delta":state.physical_command-state.previous_physical_command,"time_since_mode_change":state.time_since_mode_change_s,"ramp_progress":state.ramp_progress,"condition_index":cidx[:,None],"recipe_id":recipe[:,None],"episode_id":recipe[:,None],"control_step":torch.full((count,1),step,device=dev),"split_id":split[:,None],"teacher_source":torch.full((count,1),teacher_source,device=dev)}
   for k,v in values.items():fields[k].append(v.detach().cpu())
   sample_counter+=count
  term.external_override.zero_();term._update_command();obs=wrapped.get_observations().to(dev)
  # Establish practical stand states solely with Specialist S.
  for step in range(150):
   state.advance(torch.zeros(n,3,device=dev),torch.ones(n,device=dev),DT)
   with torch.inference_mode():action=stop(obs["policy"],gait)
   if step in (100,110,120,130,140,149):record("STAND_HOLD","STAND_HOLD",action,0,-150+step)
   obs,_,_,_=wrapped.step(action);obs=obs.to(dev)
  # Same robot state and old 124D prefix; target mode and teacher action differ.
  with torch.inference_mode():stand_b0=stop(obs["policy"],gait);walk_b0=walk(obs["policy"],gait)
  record("STAND_HOLD","STAND_HOLD",stand_b0,0,0)
  state.request(torch.full((n,),int(MotionMode.WALK),device=dev))
  state.ramp_progress.zero_();record("STAND_TO_WALK_B0","STAND_TO_WALK",walk_b0,1,0)
  previous=torch.zeros(n,3,device=dev)
  ramp_record={1,2,3,4,5,7,10,15,20,30,45,60,74}
  acquisition_record={75,76,77,78,79,80,82,85,90,100,110,120}
  steady_record={130,140,150,165,180,200,225}
  for step in range(226):
   progress=torch.tensor(min(1.,step/RAMP),device=dev).expand(n);physical=target*minimum_jerk(progress)[:,None]
   state.advance(physical,progress,0.0 if step==0 else DT);env_command(physical);obs=wrapped.get_observations().to(dev)
   with torch.inference_mode():action=walk(obs["policy"],gait)
   if step in ramp_record:record("STAND_TO_WALK_RAMP","STAND_TO_WALK",action,1,step)
   elif step in acquisition_record:record("STAND_TO_WALK_ACQUISITION","STAND_TO_WALK",action,1,step)
   elif step in steady_record:
    # Context differs by registered condition; teacher remains W for all.
    context=torch.tensor([4 if kinds[int(ci)]=="walk" else (5 if kinds[int(ci)]=="pure_yaw" else 6) for ci in cidx],device=dev)
    x=build_observation_141(obs["policy"],state);values={"observation_141":x,"teacher_action":action,"context":context[:,None],"group":torch.where(context[:,None]==4,torch.full((n,1),2,device=dev),torch.full((n,1),3,device=dev)),"physical_command":state.physical_command,"target_mode":state.target_mode[:,None],"previous_target_mode":state.previous_target_mode[:,None],"previous_physical_command":state.previous_physical_command,"command_delta":state.physical_command-state.previous_physical_command,"time_since_mode_change":state.time_since_mode_change_s,"ramp_progress":state.ramp_progress,"condition_index":cidx[:,None],"recipe_id":recipe[:,None],"episode_id":recipe[:,None],"control_step":torch.full((n,1),step,device=dev),"split_id":split[:,None],"teacher_source":torch.ones(n,1,device=dev)}
    for k,v in values.items():fields[k].append(v.detach().cpu())
   previous=physical;obs,_,_,_=wrapped.step(action);obs=obs.to(dev)
  # STOP request changes mode immediately; physical command then minimum-jerk ramps down.
  state.request(torch.full((n,),int(MotionMode.STAND),device=dev));stop_record={0,1,2,3,4,5,7,10,15,20,30,45,60,74}
  for step in range(150):
   progress=torch.tensor(min(1.,step/RAMP),device=dev).expand(n);physical=target*(1.-minimum_jerk(progress))[:,None]
   state.advance(physical,progress,0.0 if step==0 else DT);env_command(physical);obs=wrapped.get_observations().to(dev)
   with torch.inference_mode():action=stop(obs["policy"],gait)
   if step in stop_record:record("WALK_TO_STAND_DECELERATION","WALK_TO_STAND",action,0,226+step)
   elif step in (75,76,77,78,80,85,90,100,110):record("WALK_TO_STAND_RECOVERY","WALK_TO_STAND",action,0,226+step)
   elif step in (120,130,140,149):record("STAND_AFTER_STOP","STAND_AFTER_STOP",action,0,226+step)
   obs,_,_,_=wrapped.step(action);obs=obs.to(dev)
  data={k:torch.cat(v,0) for k,v in fields.items()};data["sample_id"]=torch.arange(len(data["observation_141"]),dtype=torch.long)+args.batch*10**9;data["dataset_name"]="Exp014StandOmniWalkTrajectoryDatasetV1";data["contexts"]=CONTEXTS;data["groups"]=GROUPS;data["conditions"]=conds;data["control_dt"]=DT
  path=RAW/f"phase1_batch_{args.batch:02d}.pt";torch.save(data,path)
  meta={"path":path.relative_to(REPO).as_posix(),"batch":args.batch,"episodes":n,"samples":len(data["sample_id"]),"split_counts":{name:int((data["split_id"]==i).sum()) for i,name in enumerate(("train","validation","held-out"))},"teacher_counts":{"S":int((data["teacher_source"]==0).sum()),"W":int((data["teacher_source"]==1).sum())},"timestamp":datetime.now(timezone(timedelta(hours=9))).isoformat()};print(json.dumps(meta,indent=2));wrapped.close()

if __name__=="__main__":main()
