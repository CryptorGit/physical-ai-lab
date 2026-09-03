"""Supplement D17 with exact 123D observation geometry and distinct P3 action."""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import gymnasium as gym
import torch

HERE=Path(__file__).resolve();REPO=HERE.parents[4]
def mod(name,path):s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d17=mod("d17base",HERE.parent/"run_phase2_d17_audit.py");d16=d17.d16;d3=d17.d3;d6=d17.d6
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli

def main():
 p=argparse.ArgumentParser();add_launcher_args(p);a,h=setup_preset_cli(p);sys.argv=[sys.argv[0],*h];cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=64;cfg.seed=20279402;cfg.episode_length_s=20.;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None
 if a.device:cfg.sim.device=agent.device=a.device
 train=torch.load(d17.D16/"raw/train_start_snapshots.pt",map_location="cpu",weights_only=False);cp=torch.load(d17.D16/"raw/checkpoints/model_040.pt",map_location="cpu",weights_only=False);d6pool=torch.load(d17.D6/"raw/snapshots/selected/train_batch_00.pt",map_location="cpu",weights_only=False);picks=[i for i,x in enumerate(train["valid"]) if x][:64]
 with launch_simulation(cfg,a):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=d16.StartWorld(wrapped,d3.load_resets(),train);policy=d16.StartPolicy(cp["residual_bound"]).to(world.device);policy.residual.load_state_dict(cp["residual_state_dict"]);policy.eval()
  active=[i for i,ok in enumerate(d6pool["w_move_acquired"]) if ok][:64];snap={k:v[active].to(world.device) for k,v in d6pool["snapshot"].items()};world.restore_snapshot(snap);gait=torch.zeros(64,device=world.device);obsref=[];phys=[];acts=[]
  for _ in range(160):
   cmd=torch.zeros(64,3,device=world.device);cmd[:,0]=.3;d6.set_command(world,cmd);obs=world.env.observation_manager.compute()["policy"]
   with torch.inference_mode():act=policy.base(obs[:,:123],gait)
   world.wrapped.step(act);obsref.append(world.env.observation_manager.compute()["policy"][:,:123].cpu());phys.append(d17.physical_features(world,64).cpu());acts.append(act.cpu())
  obsref=torch.cat(obsref).to(world.device);phys=torch.cat(phys).to(world.device);acts=torch.cat(acts).to(world.device);d17.restore_source(world,train,picks);source_obs=world.env.observation_manager.compute()["policy"][:,:123];mean=obsref.mean(0);std=obsref.std(0).clamp_min(1e-4);od,oi=d17.nearest_distance(source_obs,obsref,mean,std);pmean=phys.mean(0);pstd=phys.std(0).clamp_min(1e-4);diag=phys[::5];diagact=acts[::5]
  previous={name:{"B0":d17.rollout(world,policy,train,picks,diag,pmean,pstd,use_r40=False,previous=name,basin_current_actions=diagact),"R40":d17.rollout(world,policy,train,picks,diag,pmean,pstd,previous=name,basin_current_actions=diagact)} for name in ("P0","P1","P2","P3")}
  out={"full_123d_observation":{"source_count":64,"basin_count":len(obsref),"nearest_distance_mean":float(od.mean()),"nearest_distance_p95":float(torch.quantile(od,.95)),"nearest_indices":oi.cpu().tolist()},"previous_action":previous,"P2_P3_distinct":True,"persistent_policy_update":0}
  d17.dump(d17.RAW/"geometry_prev_supplement.json",out);print(json.dumps({"obs123_mean":out["full_123d_observation"]["nearest_distance_mean"],"P3_R40":previous["P3"]["R40"]["acquisition"]},indent=2));wrapped.close()
if __name__=="__main__":main()
