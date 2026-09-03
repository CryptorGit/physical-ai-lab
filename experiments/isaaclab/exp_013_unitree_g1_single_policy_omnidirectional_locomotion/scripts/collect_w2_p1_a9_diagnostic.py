"""Collect the exact-control-step A9 diagnostic trajectories (no training)."""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path
import gymnasium as gym
import torch

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]; EXP=HERE.parent.parent
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
OUT=BASE/"phase_w2_p1_a9_observation_history_contract_preflight"; M0=BASE/"phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight"
A8=BASE/"phase_w2_p1_a8_offline_start_teacher_oracle"; R2=BASE/"phase_w2_p1_a7_r2_rear_yaw_start_teacher_replay_v2/checkpoints"
STOP=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
W1B=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
sys.path[:0]=[str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),str(EXP/"src")]
import isaaclab_tasks  # noqa:F401,E402
import g1_omnidirectional.tasks  # noqa:F401,E402
from g1_omnidirectional.policy import FrozenGaitActor  # noqa:E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa:E402
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli  # noqa:E402

p=argparse.ArgumentParser();p.add_argument("--mode",choices=("formal","local"),required=True);p.add_argument("--batch",type=int,default=4);add_launcher_args(p);args,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0],*hydra]
N=1024;RAMP=75;HOLD=200;DT=.02
def mj(x):x=x.clamp(0,1);return 10*x**3-15*x**4+6*x**5
def sh(x):return hashlib.sha256(x).hexdigest()
def split_ids(ids):
 g=torch.Generator().manual_seed(20278721);order=ids[torch.randperm(len(ids),generator=g)];n=len(order);return {"train":order[:round(.70*n)],"validation":order[round(.70*n):round(.85*n)],"heldout":order[round(.85*n):]}
def canonical_condition(d,y):return f"D{int(round(d))%360:03d}_Y{y:+.1f}"
def main():
 cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=N;cfg.episode_length_s=14.;cfg.seed=20278501;cfg.observations.policy.enable_corruption=False
 if args.device:cfg.sim.device=agent.device=args.device
 masks=json.loads((M0/"a7_environment_masks.json").read_text())["batches"][str(args.batch)];accepted=torch.nonzero(torch.tensor(masks["accepted_mask"],dtype=torch.bool)).flatten();splits=split_ids(accepted)
 split_of=torch.full((N,),-1,dtype=torch.long)
 for si,name in enumerate(("train","validation","heldout")):split_of[splits[name]]=si
 cmap=json.loads((A8/"offline_start_teacher_condition_map_v1.json").read_text())["condition_map"];formal={r["condition_id"]:r["selected_checkpoint_update"] for r in cmap}
 local=json.loads((OUT/"local_command_labelability.json").read_text())["rows"]
 points=[]
 for r in local:
  if not r["labelable"]:continue
  source=next((A8/"raw/local_neighborhood"/r["formal_boundary_condition"]).glob("condition_*.json"));upd=int(Path(json.loads(source.read_text())["policy"]).stem.split("_")[-1])
  points.append((r["point_id"],float(r["direction"]),float(r["speed"]),float(r["yaw"]),upd,int(r["direction_delta"]),float(r["yaw_delta"])))
 cfg.scene.num_envs=N
 raw=OUT/"observation_history_diagnostic_dataset";raw.mkdir(parents=True,exist_ok=True)
 with launch_simulation(cfg,args):
  w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);env=w.unwrapped;dev=env.device;cmd=env.command_manager.get_term("base_velocity");cmd.external_override_enabled=True;stop=FrozenGaitActor(STOP).to(dev).eval();gait=torch.zeros(N,device=dev);allids=torch.arange(N,device=dev)
  sensor=env.scene["contact_forces"];feet=sensor.find_bodies(".*_ankle_roll_link")[0]
  for _ in range(args.batch+1):
   env.reset(env_ids=allids);cmd.external_override.zero_();cmd._update_command();obs=w.get_observations().to(dev)
   for _ in range(150):
    with torch.inference_mode():a=stop(obs["policy"],gait)
    obs,_,_,_=w.step(a);obs=obs.to(dev)
  # Replay V2 invariant: diagnostic policies are constructed only after identity roll-in.
  p10=FrozenGaitActor(R2/"model_010.pt").to(dev).eval();p150=FrozenGaitActor(R2/"model_150.pt").to(dev).eval();moving=FrozenGaitActor(W1B).to(dev).eval()
  target=torch.zeros(N,4,device=dev);updates=torch.zeros(N,dtype=torch.long,device=dev);point_ids=[""]*N
  for si,name in enumerate(("train","validation","heldout")):
   ids=splits[name];
   if args.mode=="formal":
    for j,e in enumerate(ids.tolist()):
     c=j%24;d=(c//3)*45.;y=(-.3,0.,.3)[c%3];rad=math.radians(d);target[e,:3]=torch.tensor([.3*math.cos(rad),.3*math.sin(rad),y],device=dev);updates[e]=formal[canonical_condition(d,y)];point_ids[e]=canonical_condition(d,y)
   else:
    perturbations={"train":((0,0.),(-5,-.03),(5,.03),(-5,.03),(5,-.03)),"validation":((0,-.03),(-5,0.)),"heldout":((0,.03),(5,0.))}[name]
    allowed=[q for q in points if (q[5],q[6]) in perturbations]
    for j,e in enumerate(ids.tolist()):
     q=allowed[j%len(allowed)];_,d,s,y,u,_,_=q;rad=math.radians(d);target[e,:3]=torch.tensor([s*math.cos(rad),s*math.sin(rad),y],device=dev);updates[e]=u;point_ids[e]=q[0]
  active=(split_of>=0).to(dev);target=target.to(dev);updates=updates.to(dev);split_dev=split_of.to(dev)
  fields={k:[] for k in ("observation","physical_command","actor_command","previous_physical_command","command_delta","time_since_command_change","ramp_progress","contact","air_time","support_phase","observation_history_8","physical_command_history_8","previous_action","teacher_action","teacher_source_update","context","condition_index","episode_id","recipe_id","control_step","split_id")}
  hist_obs=[];hist_cmd=[];air=torch.zeros(N,2,device=dev);prev_phys=torch.zeros(N,4,device=dev)
  def obs124():return torch.cat((obs["policy"],gait[:,None]),1)
  def tick_history(physical):
   nonlocal hist_obs,hist_cmd,air
   contact=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1)>5;air=torch.where(contact,torch.zeros_like(air),air+DT);hist_obs=(hist_obs+[obs124().clone()])[-8:];hist_cmd=(hist_cmd+[physical.clone()])[-8:]
   while len(hist_obs)<8:hist_obs.insert(0,hist_obs[0].clone());hist_cmd.insert(0,hist_cmd[0].clone())
   return contact
  def record(context,step,physical,contact,label,source):
   ids=torch.nonzero(active).flatten();actor=physical.clone();actor[:,2]=torch.where(actor[:,2]>0,1.5*actor[:,2],actor[:,2]);delta=physical-prev_phys;support=torch.stack((contact.all(1),contact[:,0]&~contact[:,1],contact[:,1]&~contact[:,0],~contact.any(1)),1)
   context_tensor=context[:,None] if torch.is_tensor(context) else torch.full((N,1),context,device=dev)
   vals={"observation":obs124(),"physical_command":physical,"actor_command":actor,"previous_physical_command":prev_phys,"command_delta":delta,"time_since_command_change":torch.full((N,1),max(0,step)*DT/5,device=dev).clamp_max(1),"ramp_progress":torch.full((N,1),max(0,min(1,step/RAMP)),device=dev),"contact":contact.float(),"air_time":air,"support_phase":support.float(),"observation_history_8":torch.stack(hist_obs,1),"physical_command_history_8":torch.stack(hist_cmd,1),"previous_action":obs124()[:,86:123],"teacher_action":label,"teacher_source_update":source[:,None].float(),"context":context_tensor,"condition_index":torch.arange(N,device=dev)[:,None],"episode_id":torch.arange(N,device=dev)[:,None],"recipe_id":torch.arange(args.batch*N,(args.batch+1)*N,device=dev)[:,None],"control_step":torch.full((N,1),step,device=dev),"split_id":split_dev[:,None]}
   for k,v in vals.items():fields[k].append(v[ids].cpu())
  # B0 uses the stop label; duplicate contexts are intentional semantic audit samples.
  physical=torch.zeros(N,4,device=dev);contact=tick_history(physical)
  with torch.inference_mode():sl=stop(obs["policy"],gait)
  record(0,-1,physical,contact,sl,torch.full((N,),-12,device=dev));record(1,0,physical,contact,sl,torch.full((N,),-12,device=dev))
  start_record={1,2,3,4,5,6,7,8,15,30,45,60,74,75,80,90,105,120,149,175,200,250,274}
  for step in range(RAMP+HOLD):
   scale=mj(torch.tensor(step/RAMP,device=dev));physical=target*scale;actor=physical.clone();actor[:,2]=torch.where(actor[:,2]>0,1.5*actor[:,2],actor[:,2]);cmd.external_override.zero_();cmd.external_override[active,:3]=actor[active,:3];cmd._update_command();obs=w.get_observations().to(dev);contact=tick_history(physical)
   with torch.inference_mode():a10=p10(obs["policy"],gait);a150=p150(obs["policy"],gait);ma=moving(obs["policy"],gait);ha=stop(obs["policy"],gait);start_label=torch.where((updates==10)[:,None],a10,a150);runtime=torch.where(active[:,None],start_label,ha)
   if step in start_record:
    context=2 if step<38 else (3 if step<75 else (4 if step<175 else torch.where(target[:,2].abs()>.01,torch.full_like(updates,6),torch.full_like(updates,5))))
    label=start_label if step<175 else ma;source=updates if step<175 else torch.full_like(updates,-200);record(context,step,physical,contact,label,source)
   prev_phys=physical.clone();obs,_,_,_=w.step(runtime);obs=obs.to(dev)
  # Stop recovery from the visited moving state.
  for step in range(40):
   physical=torch.zeros(N,4,device=dev);cmd.external_override.zero_();cmd._update_command();obs=w.get_observations().to(dev);contact=tick_history(physical)
   with torch.inference_mode():ha=stop(obs["policy"],gait)
   if step in (0,1,2,3,4,5,7,10,15,20,30,39):record(7,275+step,physical,contact,ha,torch.full_like(updates,-12))
   prev_phys=physical.clone();obs,_,_,_=w.step(ha);obs=obs.to(dev)
  data={k:torch.cat(v) for k,v in fields.items()};data["point_ids"]=point_ids;data["mode"]=args.mode;data["batch"]=args.batch;data["control_dt"]=DT
  path=raw/f"{args.mode}_batch_{args.batch}.pt";torch.save(data,path);print(json.dumps({"path":str(path),"samples":len(data["observation"]),"split_counts":{n:int((data["split_id"]==i).sum()) for i,n in enumerate(("train","validation","heldout"))}}));w.close()
if __name__=="__main__":main()
