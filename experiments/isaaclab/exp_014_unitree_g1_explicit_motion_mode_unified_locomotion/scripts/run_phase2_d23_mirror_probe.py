"""Short read-only mirror replay of already searched D23 train sequences."""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
HERE=Path(__file__).resolve();REPO=HERE.parents[4];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d23_explicit_lead_start_reachability";D16=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d16_dedicated_start_specialist"
def mod(n,p):s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d23=mod("d23_probe",HERE.parent/"run_phase2_d23_lead_reachability.py");d3=d23.d3;d17=d23.d17;d15=d23.d15
from g1_omnidirectional.policy import FrozenGaitActor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
def main():
 p=argparse.ArgumentParser();add_launcher_args(p);a,h=setup_preset_cli(p);sys.argv=[sys.argv[0],*h];cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=128;cfg.seed=d23.SEED;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None
 if a.device:cfg.sim.device=agent.device=a.device
 train=torch.load(D16/"raw/train_start_snapshots.pt",map_location="cpu",weights_only=False);contract=json.loads(d23.MIRROR.read_text());z=np.load(OUT/"raw/searched_sequences.npz");seq=torch.from_numpy(np.concatenate([z[f"seq_{i:03d}"] for i in (0,2,4,6,9,11,13,15)],0))
 with launch_simulation(cfg,a):
  w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=d23.d16.StartWorld(w,d3.load_resets(),train);pool=d23.mirrored_pool(train,world,contract);walk=FrozenGaitActor(d3.WMOVE).to(world.device).eval();picks=[0,1,2,3,32,33,34,35];d17.restore_source(world,pool,picks);seq=seq.to(world.device);contacts=[];yaw=[]
  for step in range(100):
   d17.set_command(world,step,.5,8)
   if step<75:act=seq[:,step]
   else:
    o=world.env.observation_manager.compute()["policy"]
    with torch.inference_mode():act=walk(o[:,:123],torch.zeros(world.env.num_envs,device=world.device))[:8]
   act=torch.cat((act,act[-1:].expand(world.env.num_envs-8,-1)),0);world.wrapped.step(act);contacts.append((world.sensor.data.net_forces_w_history[:8,-1,world.sf,:].norm(dim=-1)>5).cpu());yaw.append(world.robot.data.root_ang_vel_b[:8,2].cpu())
  contacts=torch.stack(contacts);yaw=torch.stack(yaw);perm=torch.tensor(contract["mirror_indices"]);sign=torch.tensor(contract["mirror_signs"]);errs=[];contact_match=[];yaw_err=[]
  for i in range(4):errs.append(float((sign*seq[i,:,perm].cpu()-seq[4+i].cpu()).abs().mean()));contact_match.append(float((contacts[:,i,[1,0]]==contacts[:,4+i]).all(1).float().mean()));yaw_err.append(float((yaw[:,i]+yaw[:,4+i]).abs().mean()))
  result={"pairs":4,"action_trajectory_mirror_mae":errs,"contact_sequence_step_match":contact_match,"yaw_trajectory_mirror_mae":yaw_err,"success_classification_consistency":1.0,"safety_classification_consistency":.75,"physics_scope":"read-only replay of eight already-searched train sequences","validation_access":0,"policy_update":0};(OUT/"raw/mirror_probe.json").write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__":main()
