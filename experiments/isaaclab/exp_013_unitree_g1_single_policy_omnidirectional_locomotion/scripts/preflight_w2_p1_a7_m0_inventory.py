"""Fresh full-batch replay inventory for A7-M0 (no policy training)."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import gymnasium as gym
import torch

HERE=Path(__file__).resolve();REPO=HERE.parents[4];EXP=HERE.parent.parent
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight"
S0=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_a7_s0_formal_stop_state_pool"
TEACHER=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
sys.path[:0]=[str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),str(EXP/"src")]
import isaaclab_tasks,g1_omnidirectional.tasks
from g1_omnidirectional.policy import FrozenGaitActor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
p=argparse.ArgumentParser();add_launcher_args(p);a,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0],*hydra]
N=1024;SEED=20278501;ROLL=150;WINDOW=100;TARGET=6144
def hs(parts):
 h=hashlib.sha256()
 for k,v in sorted(parts.items()):
  t=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
 return h.hexdigest()
def per_state(parts,i):return hs({k:v[i] for k,v in parts.items()})
def support(c):
 o=torch.full((len(c),),4,device=c.device,dtype=torch.long);o[~c.any(1)]=3;o[c.all(1)]=2;o[c[:,0]&~c[:,1]]=0;o[c[:,1]&~c[:,0]]=1;return o
cfg,ac=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=N;cfg.episode_length_s=12.;cfg.seed=SEED;cfg.observations.policy.enable_corruption=False
if a.device:cfg.sim.device=ac.device=a.device
rows=[];batches=[];selected=0
with launch_simulation(cfg,a):
 w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=ac.clip_actions);env=w.unwrapped;robot=env.scene["robot"];sensor=env.scene["contact_forces"];feet=sensor.find_bodies(".*_ankle_roll_link")[0];rfeet=robot.find_bodies(".*_ankle_roll_link")[0];term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True;teacher=FrozenGaitActor(TEACHER).to(env.device).eval();g=torch.zeros(N,device=env.device);limits=robot.data.joint_vel_limits;limits=limits[...,1].abs() if limits.ndim==3 else limits
 for batch in range(7):
  ids=torch.arange(N,device=env.device);env.reset(env_ids=ids);term.external_override.zero_();term._update_command();obs=w.get_observations().to(env.device);speed=torch.zeros(N,device=env.device);yaw=torch.zeros(N,device=env.device);fall=torch.zeros(N,dtype=torch.bool,device=env.device);slip=fall.clone();impact=fall.clone();sat=fall.clone();ss=torch.zeros(N,dtype=torch.long,device=env.device);vs=ss.clone()
  for step in range(ROLL):
   with torch.inference_mode():act=teacher(obs["policy"],g)
   prev=env.action_manager.prev_action.clone();obs,_,done,extra=w.step(act);obs=obs.to(env.device);timeout=extra.get("time_outs",torch.zeros_like(done)).bool();fall|=done.bool()&~timeout;force=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);contact=force>5;fs=torch.linalg.vector_norm(robot.data.body_lin_vel_w[:,rfeet,:2],dim=-1);bad=((fs>.55)&contact).any(1);ss=torch.where(bad,ss+1,torch.zeros_like(ss));slip|=ss>=5;impact|=force.amax(1)>3500;ratio=robot.data.joint_vel.abs().div(limits.clamp_min(1e-6)).amax(1);vs=torch.where(ratio>.95,vs+1,torch.zeros_like(vs));sat|=vs>=5
   if step>=ROLL-WINDOW:speed+=torch.linalg.vector_norm(robot.data.root_lin_vel_b[:,:2],dim=-1);yaw+=robot.data.root_ang_vel_b[:,2].abs()
  speed/=WINDOW;yaw/=WINDOW;ok=(speed<=.08)&(yaw<=.08)&~fall&~slip&~impact&~sat;cf=sensor.data.net_forces_w_history[:,-1,feet,:].clone();ct=cf.norm(dim=-1)>5
  parts={"root_pos_w":robot.data.root_pos_w.cpu(),"root_quat_w":robot.data.root_quat_w.cpu(),"root_lin_vel_w":robot.data.root_lin_vel_w.cpu(),"root_ang_vel_w":robot.data.root_ang_vel_w.cpu(),"joint_pos":robot.data.joint_pos.cpu(),"joint_vel":robot.data.joint_vel.cpu(),"body_state_w":robot.data.body_state_w.cpu(),"current_action":act.cpu(),"previous_action":prev.cpu(),"policy_observation":obs["policy"].cpu(),"contact_force":cf.cpu(),"contact_flags":ct.cpu(),"support_state":support(ct).cpu()}
  take=max(0,min(int(ok.sum()),TARGET-selected));accepted_seen=0;reason_counts={}
  for i in range(N):
   accepted=bool(ok[i]);chosen=accepted and accepted_seen<take
   if accepted:accepted_seen+=1
   sid=batch*N+i;split="none"
   if chosen:
    pos=selected+accepted_seen-1
    split="train" if pos<4096 else "validation" if pos<5120 else "heldout"
   reasons=[]
   if speed[i]>.08:reasons.append("speed")
   if yaw[i]>.08:reasons.append("yaw")
   if fall[i]:reasons.append("fall")
   if slip[i]:reasons.append("slip")
   if impact[i]:reasons.append("impact")
   if sat[i]:reasons.append("saturation")
   reason="PASS" if accepted else "+".join(reasons);reason_counts[reason]=reason_counts.get(reason,0)+1
   rows.append({"source_batch_id":batch,"environment_index":i,"state_id":sid,"accepted":accepted,"selected_pool":chosen,"split":split,"rejection_reason":reason,"semantic_state_hash":per_state(parts,i),"mean_speed":float(speed[i]),"mean_abs_yaw":float(yaw[i])})
  chosen_idx=torch.nonzero(ok.cpu()).flatten()[:take];selected_parts={k:v[chosen_idx] for k,v in parts.items()};batches.append({"source_batch_id":batch,"full_batch_semantic_hash":hs(parts),"selected_semantic_hash_m0_schema":hs(selected_parts),"accepted":int(ok.sum()),"selected":take,"rejected":int((~ok).sum()),"reason_counts":reason_counts});selected+=take
 w.close()
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/"raw_source_batch_inventory.csv").open("w",newline="",encoding="utf-8") as f:wr=csv.DictWriter(f,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)
 (OUT/"raw_full_batch_replay.json").write_text(json.dumps({"batches":batches,"selected":selected},indent=2,sort_keys=True)+"\n",encoding="utf-8")
 print(json.dumps({"selected":selected,"batches":batches},indent=2))
