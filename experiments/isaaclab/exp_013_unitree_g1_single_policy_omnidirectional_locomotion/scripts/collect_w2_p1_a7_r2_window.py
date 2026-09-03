"""Collect one A7-R2 masked window after a canonical V2 stop replay."""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path
import gymnasium as gym
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve();REPO=HERE.parents[4];EXP=HERE.parent.parent;BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion";M0=BASE/"phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight";TEACHER=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
sys.path[:0]=[str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),str(EXP/"src")]
import isaaclab_tasks,g1_omnidirectional.tasks
from g1_omnidirectional.policy import FrozenGaitActor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
p=argparse.ArgumentParser();p.add_argument("--policy",required=True);p.add_argument("--targets",required=True);p.add_argument("--output",required=True);p.add_argument("--batch",type=int,required=True);p.add_argument("--offset",type=int,required=True);p.add_argument("--noise-seed",type=int,required=True);add_launcher_args(p);a,h=setup_preset_cli(p);sys.argv=[sys.argv[0],*h]
N=1024;ROLL=150;T=24
def semantic(parts):
 q=hashlib.sha256()
 for k,v in sorted(parts.items()):t=v.detach().cpu().contiguous();q.update(k.encode());q.update(str(t.dtype).encode());q.update(str(tuple(t.shape)).encode());q.update(t.numpy().tobytes())
 return q.hexdigest()
def support(c):
 o=torch.full((len(c),),4,device=c.device,dtype=torch.long);o[~c.any(1)]=3;o[c.all(1)]=2;o[c[:,0]&~c[:,1]]=0;o[c[:,1]&~c[:,0]]=1;return o
def mj(x):x=x.clamp(0.,1.);return 10*x**3-15*x**4+6*x**5
def critic(s,o):
 x=F.elu(F.linear(o,s["mlp.0.weight"],s["mlp.0.bias"]));x=F.elu(F.linear(x,s["mlp.2.weight"],s["mlp.2.bias"]));x=F.elu(F.linear(x,s["mlp.4.weight"],s["mlp.4.bias"]));return F.linear(x,s["mlp.6.weight"],s["mlp.6.bias"]).squeeze(-1)
cfg,ac=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=N;cfg.episode_length_s=12.;cfg.seed=20278501;cfg.observations.policy.enable_corruption=False
if a.device:cfg.sim.device=ac.device=a.device
masks=json.loads((M0/"a7_environment_masks.json").read_text())["batches"][str(a.batch)];train_cpu=torch.tensor(masks["train_mask"],dtype=torch.bool);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
with launch_simulation(cfg,a):
 w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=ac.clip_actions);env=w.unwrapped;robot=env.scene["robot"];sensor=env.scene["contact_forces"];feet=sensor.find_bodies(".*_ankle_roll_link")[0];rfeet=robot.find_bodies(".*_ankle_roll_link")[0];term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True;teacher=FrozenGaitActor(TEACHER).to(env.device).eval();g=torch.zeros(N,device=env.device);ids=torch.arange(N,device=env.device);limits=robot.data.joint_vel_limits;limits=limits[...,1].abs() if limits.ndim==3 else limits
 # Replay source batches in canonical order with no trainable policy objects resident.
 for batch_id in range(a.batch+1):
  env.reset(env_ids=ids);term.external_override.zero_();term._update_command();obs=w.get_observations().to(env.device);speed=torch.zeros(N,device=env.device);yaw=torch.zeros(N,device=env.device);fall=torch.zeros(N,dtype=torch.bool,device=env.device);slip=fall.clone();impact=fall.clone();sat=fall.clone();ss=torch.zeros(N,dtype=torch.long,device=env.device);vs=ss.clone()
  for step in range(ROLL):
   with torch.inference_mode():act=teacher(obs["policy"],g)
   prev=env.action_manager.prev_action.clone();obs,_,done,extra=w.step(act);obs=obs.to(env.device);timeout=extra.get("time_outs",torch.zeros_like(done)).bool();fall|=done.bool()&~timeout;force=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);contact=force>5;fs=torch.linalg.vector_norm(robot.data.body_lin_vel_w[:,rfeet,:2],dim=-1);bad=((fs>.55)&contact).any(1);ss=torch.where(bad,ss+1,torch.zeros_like(ss));slip|=ss>=5;impact|=force.amax(1)>3500;ratio=robot.data.joint_vel.abs().div(limits.clamp_min(1e-6)).amax(1);vs=torch.where(ratio>.95,vs+1,torch.zeros_like(vs));sat|=vs>=5
   if step>=50:speed+=torch.linalg.vector_norm(robot.data.root_lin_vel_b[:,:2],dim=-1);yaw+=robot.data.root_ang_vel_b[:,2].abs()
 accepted=(speed/100<=.08)&(yaw/100<=.08)&~fall&~slip&~impact&~sat;expected=torch.tensor(masks["accepted_mask"],device=env.device)
 if not torch.equal(accepted,expected):raise RuntimeError("EXP013_W2_P1_A7_R2_REPLAY_V2_IDENTITY_FAIL")
 cf=sensor.data.net_forces_w_history[:,-1,feet,:].clone();ct=cf.norm(dim=-1)>5
 inventory_hash=semantic({"root_pos_w":robot.data.root_pos_w,"root_quat_w":robot.data.root_quat_w,"root_lin_vel_w":robot.data.root_lin_vel_w,"root_ang_vel_w":robot.data.root_ang_vel_w,"joint_pos":robot.data.joint_pos,"joint_vel":robot.data.joint_vel,"body_state_w":robot.data.body_state_w,"current_action":act,"previous_action":prev,"policy_observation":obs["policy"],"contact_force":cf,"contact_flags":ct,"support_state":support(ct)})
 capture_hash=semantic({"root_pos_w":robot.data.root_pos_w,"root_quat_w":robot.data.root_quat_w,"root_lin_vel_w":robot.data.root_lin_vel_w,"root_ang_vel_w":robot.data.root_ang_vel_w,"joint_pos":robot.data.joint_pos,"joint_vel":robot.data.joint_vel,"policy_observation":obs["policy"],"contact_force":cf})
 # V2 boundary: only now load the current policy payload and construct policy state.
 policy_path=Path(a.policy);payload=torch.load(policy_path,map_location="cpu",weights_only=False);policy=FrozenGaitActor(policy_path).to(env.device).eval();critic_state={k:v.to(env.device) for k,v in payload["critic_state_dict"].items()};std=payload["actor_state_dict"]["distribution.log_std_walk"].exp().to(env.device);targets=torch.load(a.targets,map_location=env.device,weights_only=True);train=train_cpu.to(env.device);gen=torch.Generator(device=env.device).manual_seed(a.noise_seed);rows={k:[] for k in ("observation","action","reward","done","old_logp","old_value","valid")};alive=train.clone()
 for step in range(a.offset+T):
  alpha=mj(torch.tensor(step/75.,device=env.device));physical=targets*alpha;actor_cmd=physical.clone();actor_cmd[:,2]=torch.where(actor_cmd[:,2]>0,actor_cmd[:,2]*1.5,actor_cmd[:,2]);term.external_override.zero_();term.external_override[train]=actor_cmd[train];term._update_command();obs=w.get_observations().to(env.device);full=torch.cat((obs["policy"],g[:,None]),1)
  with torch.inference_mode():mean=policy(obs["policy"],g);sample=mean+torch.randn(mean.shape,generator=gen,device=env.device)*std;house=teacher(obs["policy"],g);action=torch.where(train[:,None],sample,house);value=critic(critic_state,full);logp=(-.5*(((sample-mean)/std)**2+2*std.log()+math.log(2*math.pi))).sum(-1)
  if step>=a.offset:
   for k,v in (("observation",full),("action",sample),("old_logp",logp),("old_value",value),("valid",alive)):rows[k].append(v.cpu())
  obs,reward,done,_=w.step(action);obs=obs.to(env.device)
  if step>=a.offset:rows["reward"].append(reward.cpu());rows["done"].append(done.cpu())
  alive&=~done.bool()
 with torch.inference_mode():last=critic(critic_state,torch.cat((obs["policy"],g[:,None]),1)).cpu()
 result={k:torch.stack(v) for k,v in rows.items()};result.update({"last_value":last,"state_id":torch.arange(N),"train_mask":train_cpu,"inventory_schema_hash_before_policy_load":inventory_hash,"capture_schema_hash_before_policy_load":capture_hash,"policy_load_timing":"AFTER_FORMAL_STOP_ROLLIN","policy_sha256":hashlib.sha256(policy_path.read_bytes()).hexdigest(),"batch":a.batch,"offset":a.offset,"noise_seed":a.noise_seed})
 torch.save(result,out);out.with_suffix(".json").write_text(json.dumps({"inventory_hash":inventory_hash,"capture_hash":capture_hash,"valid_samples":int(result["valid"].sum()),"batch":a.batch,"offset":a.offset,"policy_load_timing":"AFTER_FORMAL_STOP_ROLLIN"},indent=2)+"\n",encoding="utf-8");print(json.dumps({"output":str(out),"valid":int(result["valid"].sum()),"batch":a.batch,"offset":a.offset}),flush=True);w.close()
