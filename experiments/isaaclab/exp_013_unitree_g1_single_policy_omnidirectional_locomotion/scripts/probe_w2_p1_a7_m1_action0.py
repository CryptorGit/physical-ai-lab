"""Compare the first safe public observation/action boundary for A7-M1."""
from __future__ import annotations
import argparse,copy,hashlib,io,json,random,sys
from pathlib import Path
from collections import OrderedDict
import gymnasium as gym,numpy as np,torch
from torch import nn
import torch.nn.functional as F
HERE=Path(__file__).resolve();REPO=HERE.parents[4];EXP=HERE.parent.parent;OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_a7_m1_full_batch_replay_identity_repair/raw";T=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt";P=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
sys.path[:0]=[str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),str(EXP/"src")]
import isaaclab_tasks,g1_omnidirectional.tasks
from g1_omnidirectional.policy import FrozenGaitActor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
p=argparse.ArgumentParser();p.add_argument('--mode',choices=('reference','production'),required=True);p.add_argument('--tag',required=True);p.add_argument('--steps',type=int,default=0);add_launcher_args(p);a,h=setup_preset_cli(p);sys.argv=[sys.argv[0],*h]
def d(x):x=x.detach().cpu().contiguous();q=hashlib.sha256();q.update(str(x.dtype).encode());q.update(str(tuple(x.shape)).encode());q.update(x.numpy().tobytes());return q.hexdigest()
class R1Actor(nn.Module):
 def __init__(self,state):
  super().__init__();self.first_base_weight=nn.Parameter(state['first_base_weight'].clone());self.first_gait_column=nn.Parameter(state['first_gait_column'].clone());self.first_bias=nn.Parameter(state['first_bias'].clone());self.hidden=nn.Sequential(nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,37));self.hidden.load_state_dict(OrderedDict((k.removeprefix('hidden.'),v) for k,v in state.items() if k.startswith('hidden.')))
 def forward(self,o,g):return self.hidden(F.linear(o,self.first_base_weight,self.first_bias)+g.reshape(-1,1)*self.first_gait_column.T)
class R1Critic(nn.Module):
 def __init__(self,state):super().__init__();self.mlp=nn.Sequential(nn.Linear(124,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,1));self.load_state_dict(state)
cfg,ac=resolve_task_config('Isaac-Exp013-G1-DirectionalBaseline-v0','rsl_rl_cfg_entry_point');cfg.scene.num_envs=1024;cfg.episode_length_s=12.;cfg.seed=20278501;cfg.observations.policy.enable_corruption=False
if a.device:cfg.sim.device=ac.device=a.device
OUT.mkdir(parents=True,exist_ok=True)
parent_pre=torch.load(P,map_location='cpu',weights_only=False) if a.mode=='production' else None
with launch_simulation(cfg,a):
 w=RslRlVecEnvWrapper(gym.make('Isaac-Exp013-G1-DirectionalBaseline-v0',cfg=cfg),clip_actions=ac.clip_actions);env=w.unwrapped;term=env.command_manager.get_term('base_velocity');term.external_override_enabled=True;teacher=FrozenGaitActor(T).to(env.device).eval();extra=None
 if a.mode=='production':
  parent=parent_pre;cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state(env.device).clone();npr=copy.deepcopy(np.random.get_state());pyr=random.getstate();policy=R1Actor(parent['actor_state_dict']).to(env.device);critic=R1Critic(parent['critic_state_dict']).to(env.device);ap=list(policy.parameters());cp=list(critic.parameters());opt=torch.optim.Adam([{'params':ap,'lr':1.5e-5,'name':'actor_mean'},{'params':cp,'lr':1.5e-5,'name':'critic'}],lr=1.5e-5);opt.load_state_dict(copy.deepcopy(parent['optimizer_state_dict']));std=parent['actor_state_dict']['distribution.log_std_walk'].exp().to(env.device);tmp=OUT/f'_temporary_r1_checkpoint_{a.tag}.pt';torch.save({'actor':policy.state_dict(),'critic':critic.state_dict(),'optimizer':opt.state_dict()},tmp);_=[d(v) for v in list(policy.state_dict().values())+list(critic.state_dict().values()) if torch.is_tensor(v)];_ += [d(v) for state in opt.state.values() for v in state.values() if torch.is_tensor(v)];torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda,env.device);np.random.set_state(copy.deepcopy(npr));random.setstate(pyr);extra=(policy,critic,opt,std)
 gait=torch.zeros(1024,device=env.device);ids=torch.arange(1024,device=env.device);env.reset(env_ids=ids);term.external_override.zero_();term._update_command();obs=w.get_observations().to(env.device)
 with torch.inference_mode():action=teacher(obs['policy'],gait)
 initial_observation=obs['policy'].clone();initial_action=action.clone()
 for _ in range(a.steps):
  obs,_,_,_=w.step(action);obs=obs.to(env.device)
  with torch.inference_mode():action=teacher(obs['policy'],gait)
 result={'mode':a.mode,'steps':a.steps,'initial_observation_hash':d(initial_observation),'initial_teacher_action_hash':d(initial_action),'observation_hash':d(obs['policy']),'teacher_action_hash':d(action),'observation_env_hashes':[d(obs['policy'][i]) for i in range(1024)],'action_env_hashes':[d(action[i]) for i in range(1024)],'production_objects_allocated':extra is not None}
 (OUT/f'action0_{a.mode}_{a.tag}.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8');w.close()
 if a.mode=='production' and tmp.exists():tmp.unlink()
