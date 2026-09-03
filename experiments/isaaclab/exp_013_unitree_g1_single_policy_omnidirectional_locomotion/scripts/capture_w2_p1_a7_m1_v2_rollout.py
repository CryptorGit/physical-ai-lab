"""V2 collector: reproduce stop state before allocating the trainable policy."""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path
import gymnasium as gym
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve();REPO=HERE.parents[4];EXP=HERE.parent.parent;BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion";OUT=BASE/"phase_w2_p1_a7_m1_full_batch_replay_identity_repair/raw";M0=BASE/"phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight";TEACHER=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt";PARENT=BASE/"phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
sys.path[:0]=[str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),str(EXP/"src")]
import isaaclab_tasks,g1_omnidirectional.tasks
from g1_omnidirectional.policy import FrozenGaitActor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
p=argparse.ArgumentParser();p.add_argument("--yaw",choices=("negative","positive"),required=True);p.add_argument("--tag",required=True);add_launcher_args(p);a,h=setup_preset_cli(p);sys.argv=[sys.argv[0],*h]
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
masks=json.loads((M0/"a7_environment_masks.json").read_text())["batches"]["0"];train_cpu=torch.tensor(masks["train_mask"],dtype=torch.bool);parent=torch.load(PARENT,map_location="cpu",weights_only=False);OUT.mkdir(parents=True,exist_ok=True)
with launch_simulation(cfg,a):
 w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=ac.clip_actions);env=w.unwrapped;robot=env.scene["robot"];sensor=env.scene["contact_forces"];feet=sensor.find_bodies(".*_ankle_roll_link")[0];term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True;teacher=FrozenGaitActor(TEACHER).to(env.device).eval();g=torch.zeros(N,device=env.device);ids=torch.arange(N,device=env.device);env.reset(env_ids=ids);term.external_override.zero_();term._update_command();obs=w.get_observations().to(env.device)
 prev=env.action_manager.prev_action.clone();act=torch.zeros(N,37,device=env.device)
 for _ in range(ROLL):
  with torch.inference_mode():act=teacher(obs["policy"],g)
  prev=env.action_manager.prev_action.clone();obs,_,_,_=w.step(act);obs=obs.to(env.device)
 cf=sensor.data.net_forces_w_history[:,-1,feet,:].clone();ct=cf.norm(dim=-1)>5
 inventory_parts={"root_pos_w":robot.data.root_pos_w,"root_quat_w":robot.data.root_quat_w,"root_lin_vel_w":robot.data.root_lin_vel_w,"root_ang_vel_w":robot.data.root_ang_vel_w,"joint_pos":robot.data.joint_pos,"joint_vel":robot.data.joint_vel,"body_state_w":robot.data.body_state_w,"current_action":act,"previous_action":prev,"policy_observation":obs["policy"],"contact_force":cf,"contact_flags":ct,"support_state":support(ct)}
 inventory_hash=semantic(inventory_parts)
 capture_parts={"root_pos_w":robot.data.root_pos_w,"root_quat_w":robot.data.root_quat_w,"root_lin_vel_w":robot.data.root_lin_vel_w,"root_ang_vel_w":robot.data.root_ang_vel_w,"joint_pos":robot.data.joint_pos,"joint_vel":robot.data.joint_vel,"policy_observation":obs["policy"],"contact_force":cf};capture_hash=semantic(capture_parts)
 # The only V2 semantic change: allocate/load policy and critic after the exact
 # formal-stop state has been generated and hashed in the same simulator.
 policy=FrozenGaitActor(PARENT).to(env.device).eval();critic_state={k:v.to(env.device) for k,v in parent["critic_state_dict"].items()};std=parent["actor_state_dict"]["distribution.log_std_walk"].exp().to(env.device);train=train_cpu.to(env.device);gen=torch.Generator(device=env.device).manual_seed(20278611);rows={k:[] for k in ("observation","action","reward","done","old_logp","old_value","valid")};alive=train.clone();yt=-.3 if a.yaw=="negative" else .3
 for step in range(T):
  alpha=mj(torch.tensor(step*env.step_dt/1.5,device=env.device));term.external_override.zero_();term.external_override[train,0]=-.3*alpha;ay=yt*alpha;ay=ay*1.5 if ay>0 else ay;term.external_override[train,2]=ay;term._update_command();obs=w.get_observations().to(env.device);full=torch.cat((obs["policy"],g[:,None]),1)
  with torch.inference_mode():mean=policy(obs["policy"],g);sample=mean+torch.randn(mean.shape,generator=gen,device=env.device)*std;house=teacher(obs["policy"],g);action=torch.where(train[:,None],sample,house);value=critic(critic_state,full);logp=(-.5*(((sample-mean)/std)**2+2*std.log()+math.log(2*math.pi))).sum(-1)
  for k,v in (("observation",full),("action",sample),("old_logp",logp),("old_value",value),("valid",alive)):rows[k].append(v.cpu())
  obs,reward,done,_=w.step(action);obs=obs.to(env.device);rows["reward"].append(reward.cpu());rows["done"].append(done.cpu());alive&=~done.bool()
 with torch.inference_mode():last=critic(critic_state,torch.cat((obs["policy"],g[:,None]),1)).cpu()
 payload={k:torch.stack(v) for k,v in rows.items()};payload.update({"last_value":last,"state_id":torch.arange(N),"train_mask":train_cpu,"inventory_schema_hash_before_policy_load":inventory_hash,"capture_schema_hash_before_policy_load":capture_hash,"policy_load_timing":"AFTER_FORMAL_STOP_ROLLIN","parent_sha256":hashlib.sha256(PARENT.read_bytes()).hexdigest(),"yaw_pass":a.yaw})
 torch.save(payload,OUT/f"v2_rollout_{a.yaw}_{a.tag}.pt");(OUT/f"v2_rollout_{a.yaw}_{a.tag}.json").write_text(json.dumps({"inventory_hash":inventory_hash,"capture_hash":capture_hash,"valid_samples":int(payload["valid"].sum()),"policy_load_timing":"AFTER_FORMAL_STOP_ROLLIN"},indent=2)+"\n",encoding="utf-8");w.close()
