"""Collect the frozen iteration-80 low-speed anchor observation dataset."""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
import gymnasium as gym
import torch

HERE=Path(__file__).resolve();EXP=HERE.parent.parent;REPO=HERE.parents[4]
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a4_low_speed_retention_consolidation"
PARENT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
sys.path.insert(0,str(EXP/"src"))
import isaaclab_tasks,g1_omnidirectional.tasks
from g1_omnidirectional.policy import FrozenGaitActor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
parser=argparse.ArgumentParser();add_launcher_args(parser);args,hydra=setup_preset_cli(parser);sys.argv=[sys.argv[0],*hydra]
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 conditions=[(d,s) for d in (i*22.5 for i in range(16)) for s in (.25,.30,.35)]
 episodes=40;count=len(conditions)*episodes
 cfg,acfg=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point")
 cfg.scene.num_envs=count;cfg.episode_length_s=9;cfg.seed=20273021;acfg.seed=cfg.seed
 if args.device:cfg.sim.device=acfg.device=args.device
 with launch_simulation(cfg,args):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=acfg.clip_actions)
  env,device=wrapped.unwrapped,wrapped.unwrapped.device;term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True
  model=FrozenGaitActor(PARENT).to(device).eval();ids=torch.arange(count,device=device)//episodes;episode=torch.arange(count,device=device)%episodes
  commands=torch.tensor([[s*math.cos(math.radians(d)),s*math.sin(math.radians(d))] for d,s in conditions],device=device)
  obs,_=wrapped.reset();obs=obs["policy"].to(device);train_obs=[];train_ref=[];hold_obs=[];hold_ref=[];steps=round(8/env.step_dt)
  for step in range(steps):
   term.external_override[:,:2]=commands[ids];term.external_override[:,2]=0
   if step==0:term._update_command();obs=wrapped.get_observations()["policy"].to(device)
   with torch.inference_mode():action=model(obs,torch.zeros(count,device=device))
   if step%50==0:
    train=episode<32;hold=~train
    train_obs.append(obs[train].cpu());train_ref.append(action[train].cpu())
    hold_obs.append(obs[hold].cpu());hold_ref.append(action[hold].cpu())
   obs,_,_,_=wrapped.step(action);obs=obs["policy"].to(device)
  cache=OUT/"low_speed_anchor_cache.pt"
  payload={"train_observations":torch.cat(train_obs),"train_reference_mean":torch.cat(train_ref),
   "holdout_observations":torch.cat(hold_obs),"holdout_reference_mean":torch.cat(hold_ref),
   "log_std_walk":model.log_std_walk.cpu(),"conditions":conditions,"samples_per_episode":len(train_obs)}
  torch.save(payload,cache)
  dump("low_speed_anchor_manifest.json",{"reference_checkpoint":str(PARENT),"reference_sha256":sha(PARENT),
   "directions":16,"speeds":[.25,.30,.35],"conditions":48,"episodes_per_condition":40,
   "duration_s":8,"deterministic":True,"total_episodes":1920,"samples_per_episode":len(train_obs),
   "train_observations":len(payload["train_observations"]),"holdout_observations":len(payload["holdout_observations"])})
  dump("low_speed_anchor_split.json",{"method":"episode-stratified by every direction/speed condition",
   "train_episodes_per_condition":32,"holdout_episodes_per_condition":8,"train_fraction":.8,"holdout_fraction":.2,
   "direction_speed_equal_weight":True})
  dump("low_speed_anchor_hashes.json",{"cache_path":str(cache),"cache_sha256":sha(cache),
   "reference_checkpoint_sha256":sha(PARENT),"immutable_after_collection":True})
  wrapped.close()
if __name__=="__main__":main()
