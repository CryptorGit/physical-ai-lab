"""Build and validate Exp013FormalStopStatePoolV1 without policy training."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,sys
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch

HERE=Path(__file__).resolve();REPO=HERE.parents[4];EXP=HERE.parent.parent
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion";OUT=BASE/"phase_w2_p1_a7_s0_formal_stop_state_pool";POOL=OUT/"formal_stop_state_pool_v1"
TEACHER=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
sys.path[:0]=[str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),str(EXP/"src")]
import isaaclab_tasks  # noqa:F401
import g1_omnidirectional.tasks  # noqa:F401
from g1_omnidirectional.policy import FrozenGaitActor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli

parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=("generate","replay","snapshot_capture","fresh_snapshot"),required=True);parser.add_argument("--output");add_launcher_args(parser);args,hydra=setup_preset_cli(parser);sys.argv=[sys.argv[0],*hydra]
SEED=20278501;N=1024;BATCHES=8;TARGET=6144;DT=.02;ROLL=150;WINDOW=100;TEST=256
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def thash(x):return sha_bytes(x.detach().cpu().contiguous().numpy().tobytes())
def semantic(data):
 h=hashlib.sha256()
 for k,v in sorted(data.items()):
  if isinstance(v,torch.Tensor):h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def dump(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def support(contact):
 out=torch.full((len(contact),),4,device=contact.device,dtype=torch.long)
 out[~contact.any(1)]=3;out[contact.all(1)]=2
 out[contact[:,0]&~contact[:,1]]=0;out[contact[:,1]&~contact[:,0]]=1
 return out
def capture(env,w,robot,sensor,feet,term,obs,action,prev,state_ids,seeds,batch):
 contact_force=sensor.data.net_forces_w_history[:,-1,feet,:].clone();contact=contact_force.norm(dim=-1)>5
 fields={"state_id":state_ids.cpu(),"source_seed":seeds.cpu(),"source_env_id":torch.arange(N),"capture_timestep":torch.full((N,),ROLL),"batch_index":torch.full((N,),batch),"root_pos_w":robot.data.root_pos_w.clone().cpu(),"root_quat_w":robot.data.root_quat_w.clone().cpu(),"root_lin_vel_w":robot.data.root_lin_vel_w.clone().cpu(),"root_ang_vel_w":robot.data.root_ang_vel_w.clone().cpu(),"joint_pos":robot.data.joint_pos.clone().cpu(),"joint_vel":robot.data.joint_vel.clone().cpu(),"body_state_w":robot.data.body_state_w.clone().cpu(),"env_origin":env.scene.env_origins.clone().cpu(),"current_action":action.clone().cpu(),"previous_action":prev.clone().cpu(),"physical_command":torch.zeros(N,3),"actor_command":torch.zeros(N,3),"gait_command":torch.zeros(N),"episode_length":env.episode_length_buf.clone().cpu(),"policy_observation":obs["policy"].clone().cpu(),"contact_force":contact_force.cpu(),"contact_force_history":sensor.data.net_forces_w_history[:,:,feet,:].clone().cpu(),"contact_flags":contact.cpu(),"support_state":support(contact).cpu()}
 for name in ("joint_pos_target","joint_vel_target","applied_torque"):
  if hasattr(robot.data,name):fields[name]=getattr(robot.data,name).clone().cpu()
 return fields
def subset(data,ids):return {k:(v[ids] if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==N else v) for k,v in data.items()}
def restore(env,robot,term,data,count):
 ids=torch.arange(count,device=env.device);origin=env.scene.env_origins[ids];local=data["root_pos_w"][:count].to(env.device)-data["env_origin"][:count].to(env.device);pose=torch.cat((local+origin,data["root_quat_w"][:count].to(env.device)),1);vel=torch.cat((data["root_lin_vel_w"][:count],data["root_ang_vel_w"][:count]),1).to(env.device)
 robot.write_root_pose_to_sim(pose,ids);robot.write_root_velocity_to_sim(vel,ids);robot.write_joint_state_to_sim(data["joint_pos"][:count].to(env.device),data["joint_vel"][:count].to(env.device),env_ids=ids);env.action_manager._action[ids]=data["current_action"][:count].to(env.device);env.action_manager._prev_action[ids]=data["previous_action"][:count].to(env.device);env.episode_length_buf[ids]=data["episode_length"][:count].to(env.device);term.external_override[ids].zero_();term._update_command();env.sim.forward()
def compare_step(ref,cur,prefix,step,rows):
 for i in range(len(ref["state_id"])):
  rows.append({"state_id":int(ref["state_id"][i]),"split":prefix,"step":step,"root_max_diff":float((cur["root"][i]-ref["root"][step,i]).abs().max()),"joint_max_diff":float((cur["joint"][i]-ref["joint"][step,i]).abs().max()),"observation_max_diff":float((cur["obs"][i]-ref["obs"][step,i]).abs().max()),"action_max_diff":float((cur["action"][i]-ref["action"][step,i]).abs().max()),"contact_match":bool(torch.equal(cur["contact"][i],ref["contact"][step,i])),"contact_force_max_diff":float((cur["force"][i]-ref["force"][step,i]).abs().max()),"support_match":int(cur["support"][i])==int(ref["support"][step,i])})
def rollout16(w,env,robot,sensor,feet,term,teacher,count):
 ids=slice(0,count);obs=w.get_observations().to(env.device);g=torch.zeros(env.num_envs,device=env.device);out={k:[] for k in ("root","joint","obs","action","contact","force","support")}
 for _ in range(16):
  with torch.inference_mode():a=teacher(obs["policy"],g)
  obs,_,_,_=w.step(a);obs=obs.to(env.device);force=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);ct=force>5
  out["root"].append(torch.cat((robot.data.root_pos_w[ids],robot.data.root_quat_w[ids],robot.data.root_lin_vel_w[ids],robot.data.root_ang_vel_w[ids]),1).cpu());out["joint"].append(torch.cat((robot.data.joint_pos[ids],robot.data.joint_vel[ids]),1).cpu());out["obs"].append(obs["policy"][ids].cpu());out["action"].append(a[ids].cpu());out["contact"].append(ct[ids].cpu());out["force"].append(force[ids].cpu());out["support"].append(support(ct)[ids].cpu())
 return {k:torch.stack(v) for k,v in out.items()}
def run_generation(save,compare_pool=False):
 cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=N;cfg.episode_length_s=12.;cfg.seed=SEED;cfg.observations.policy.enable_corruption=False
 if args.device:cfg.sim.device=agent.device=args.device
 summary=[];hashes=[];prov=[];all_state_sem=[];accepted_ids=[];parts=[];total=0
 with launch_simulation(cfg,args):
  w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);env=w.unwrapped;robot=env.scene["robot"];sensor=env.scene["contact_forces"];feet=sensor.find_bodies(".*_ankle_roll_link")[0];rfeet=robot.find_bodies(".*_ankle_roll_link")[0];term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True;teacher=FrozenGaitActor(TEACHER).to(env.device).eval();g=torch.zeros(N,device=env.device)
  for batch in range(BATCHES):
   ids=torch.arange(N,device=env.device);env.reset(env_ids=ids);term.external_override.zero_();term._update_command();obs=w.get_observations().to(env.device);speed_sum=torch.zeros(N,device=env.device);yaw_sum=torch.zeros(N,device=env.device);fall=torch.zeros(N,dtype=torch.bool,device=env.device);slip=torch.zeros_like(fall);impact=torch.zeros_like(fall);sat_streak=torch.zeros(N,dtype=torch.long,device=env.device);sat=torch.zeros_like(fall);slip_streak=torch.zeros_like(sat_streak);switch=torch.zeros(N,device=env.device);flight=torch.zeros(N,device=env.device);dbl=torch.zeros(N,device=env.device);foot_sum=torch.zeros(N,device=env.device);last_contact=None
   reset_cpu=torch.get_rng_state().clone();reset_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
   limits=robot.data.joint_vel_limits;limits=limits[...,1].abs() if limits.ndim==3 else limits
   for step in range(ROLL):
    with torch.inference_mode():a=teacher(obs["policy"],g)
    prev=env.action_manager.prev_action.clone();obs,_,done,extras=w.step(a);obs=obs.to(env.device);timeout=extras.get("time_outs",torch.zeros_like(done)).bool();fall|=done.bool()&~timeout;force=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);ct=force>5;fs=torch.linalg.vector_norm(robot.data.body_lin_vel_w[:,rfeet,:2],dim=-1);bad=((fs>.55)&ct).any(1);slip_streak=torch.where(bad,slip_streak+1,torch.zeros_like(slip_streak));slip|=slip_streak>=5;impact|=force.amax(1)>3500
    ratio=robot.data.joint_vel.abs().div(limits.clamp_min(1e-6)).amax(1);sat_streak=torch.where(ratio>.95,sat_streak+1,torch.zeros_like(sat_streak));sat|=sat_streak>=5
    if last_contact is not None:switch+=(ct!=last_contact).any(1)
    last_contact=ct.clone();flight+=(~ct.any(1));dbl+=ct.all(1);foot_sum+=fs.mean(1)
    if step>=ROLL-WINDOW:speed_sum+=torch.linalg.vector_norm(robot.data.root_lin_vel_b[:,:2],dim=-1);yaw_sum+=robot.data.root_ang_vel_b[:,2].abs()
   mean_speed=speed_sum/WINDOW;mean_yaw=yaw_sum/WINDOW;ok=(mean_speed<=.08)&(mean_yaw<=.08)&~fall&~slip&~impact&~sat;state_ids=torch.arange(batch*N,(batch+1)*N);seeds=torch.arange(SEED+batch*N,SEED+(batch+1)*N);snap=capture(env,w,robot,sensor,feet,term,obs,a,prev,state_ids,seeds,batch);accept_idx=torch.nonzero(ok.cpu(),as_tuple=False).flatten();take=accept_idx[:max(0,TARGET-total)];chosen=subset(snap,take);sem=semantic(chosen);all_state_sem.append(sem);accepted_ids.extend(chosen["state_id"].tolist());parts.append(chosen);total+=len(take)
   summary.append({"batch":batch,"attempts":N,"accepted":int(ok.sum()),"rejected":int((~ok).sum()),"mean_speed":float(mean_speed.mean()),"p95_speed":float(torch.quantile(mean_speed,.95)),"mean_abs_yaw":float(mean_yaw.mean()),"p95_abs_yaw":float(torch.quantile(mean_yaw,.95)),"fall":float(fall.float().mean()),"slip":float(slip.float().mean()),"impact":float(impact.float().mean()),"saturation":float(sat.float().mean()),"contact_switch_rate":float((switch/3).mean()),"foot_speed":float((foot_sum/ROLL).mean()),"flight_fraction":float((flight/ROLL).mean()),"double_support_fraction":float((dbl/ROLL).mean()),"semantic_hash":sem})
   print(json.dumps({"batch":batch,"accepted":int(ok.sum()),"selected":len(take),"total_selected":total,"rejected":int((~ok).sum()),"mean_speed":float(mean_speed.mean()),"mean_yaw":float(mean_yaw.mean()),"fall":float(fall.float().mean()),"slip":float(slip.float().mean()),"impact":float(impact.float().mean()),"saturation":float(sat.float().mean())}),flush=True)
   if total>=TARGET:
    merged={k:torch.cat([p[k] for p in parts],dim=0)[:TARGET] for k in parts[0]}
    result={"summary":summary,"accepted_ids":accepted_ids,"batch_semantic_hashes":all_state_sem,"whole_pool_semantic_hash":semantic(merged)}
    if args.output:dump(Path(args.output),result)
    if save:
     for split,start,count in (("train",0,4096),("validation",4096,1024),("heldout",5120,1024)):
      d=POOL/(split+"_state_chunks");d.mkdir(parents=True,exist_ok=True)
      for ci in range(count//256):
       sl=slice(start+ci*256,start+(ci+1)*256);part={k:v[sl] for k,v in merged.items()};path=d/f"{split}_state_chunk_{ci:03d}.pt";torch.save(part,path);hashes.append({"path":str(path.relative_to(OUT)).replace("\\","/"),"byte_sha256":sha_bytes(path.read_bytes()),"semantic_sha256":semantic(part),"states":256})
      for pi in range(start,start+count):prov.append({"state_id":int(merged["state_id"][pi]),"source_seed":int(merged["source_seed"][pi]),"source_environment_id":int(merged["source_env_id"][pi]),"batch":int(merged["batch_index"][pi]),"split":split,"capture_timestep":ROLL})
     result["hashes"]=hashes;dump(OUT/"raw_generation.json",result)
     with (OUT/"raw_state_provenance.csv").open("w",newline="",encoding="utf-8") as f:wri=csv.DictWriter(f,fieldnames=prov[0]);wri.writeheader();wri.writerows(prov)
    w.close();return summary
  w.close();raise RuntimeError(f"accepted only {total} of {TARGET} after {BATCHES*N} attempts")
def snapshot_capture():
 cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=N;cfg.episode_length_s=12.;cfg.seed=SEED;cfg.observations.policy.enable_corruption=False
 if args.device:cfg.sim.device=agent.device=args.device
 with launch_simulation(cfg,args):
  w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);env=w.unwrapped;robot=env.scene["robot"];sensor=env.scene["contact_forces"];feet=sensor.find_bodies(".*_ankle_roll_link")[0];term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True;teacher=FrozenGaitActor(TEACHER).to(env.device).eval();g=torch.zeros(N,device=env.device);ids=torch.arange(N,device=env.device);env.reset(env_ids=ids);term.external_override.zero_();term._update_command();obs=w.get_observations().to(env.device)
  for _ in range(ROLL):
   with torch.inference_mode():a=teacher(obs["policy"],g)
   prev=env.action_manager.prev_action.clone();obs,_,_,_=w.step(a);obs=obs.to(env.device)
  snap=capture(env,w,robot,sensor,feet,term,obs,a,prev,torch.arange(N),torch.arange(SEED,SEED+N),0);test=subset(snap,torch.arange(768));ref=rollout16(w,env,robot,sensor,feet,term,teacher,768);restore(env,robot,term,test,768);pre=w.get_observations().to(env.device)["policy"][:768].cpu()
  with torch.inference_mode():pre_a=teacher(pre.to(env.device),torch.zeros(768,device=env.device)).cpu()
  pre_json={"observation_max_difference":float((pre-test["policy_observation"]).abs().max()),"teacher_mean_action_max_difference":float((pre_a-teacher(test["policy_observation"].to(env.device),torch.zeros(768,device=env.device)).cpu()).abs().max()),"root_joint_public_api_written":True,"contact_sensor_history_restored":False}
  same=rollout16(w,env,robot,sensor,feet,term,teacher,768);rows=[]
  for si,split in enumerate(("train","validation","heldout")):
   sl=slice(si*256,(si+1)*256);pack={k:v[sl] for k,v in test.items()};rr={k:v[:,sl] for k,v in ref.items()};cc={k:v[:,sl] for k,v in same.items()};torch.save({"snapshot":pack,"reference":rr},POOL/f"{split}_parity_reference.pt")
   for step in range(16):compare_step({"state_id":pack["state_id"],**rr},{k:v[step] for k,v in cc.items()},split,step,rows)
  dump(OUT/"raw_snapshot_pre_step.json",pre_json)
  with (OUT/"raw_same_snapshot.csv").open("w",newline="",encoding="utf-8") as f:wri=csv.DictWriter(f,fieldnames=rows[0]);wri.writeheader();wri.writerows(rows)
  w.close();return
def fresh_snapshot():
 refs=[]
 for split in ("train","validation","heldout"):refs.append(torch.load(POOL/f"{split}_parity_reference.pt",map_location="cpu",weights_only=False))
 cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=768;cfg.episode_length_s=12.;cfg.seed=SEED;cfg.observations.policy.enable_corruption=False
 if args.device:cfg.sim.device=agent.device=args.device
 rows=[]
 with launch_simulation(cfg,args):
  w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);env=w.unwrapped;robot=env.scene["robot"];sensor=env.scene["contact_forces"];feet=sensor.find_bodies(".*_ankle_roll_link")[0];term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True;teacher=FrozenGaitActor(TEACHER).to(env.device).eval();merged={k:torch.cat([r["snapshot"][k] for r in refs]) for k in refs[0]["snapshot"]};restore(env,robot,term,merged,768);rest=rollout16(w,env,robot,sensor,feet,term,teacher,768)
  for si,split in enumerate(("train","validation","heldout")):
   ref=refs[si]["reference"];cur={k:v[:,si*256:(si+1)*256] for k,v in rest.items()};refpack={"state_id":refs[si]["snapshot"]["state_id"],**ref}
   for step in range(16):compare_step(refpack,{k:v[step] for k,v in cur.items()},split,step,rows)
  with (OUT/"raw_fresh_snapshot.csv").open("w",newline="",encoding="utf-8") as f:wri=csv.DictWriter(f,fieldnames=rows[0]);wri.writeheader();wri.writerows(rows)
  w.close();return
if __name__=="__main__":
 POOL.mkdir(parents=True,exist_ok=True)
 if args.mode=="generate":run_generation(True)
 elif args.mode=="replay":run_generation(False)
 elif args.mode=="snapshot_capture":snapshot_capture()
 else:fresh_snapshot()
