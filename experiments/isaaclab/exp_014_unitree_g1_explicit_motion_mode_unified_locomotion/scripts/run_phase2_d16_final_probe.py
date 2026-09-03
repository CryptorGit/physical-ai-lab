"""Read-only C1 stopping-point probe for D16 residual statistics."""
from __future__ import annotations
import argparse,importlib.util,json,sys
from pathlib import Path
import gymnasium as gym
import torch
HERE=Path(__file__).resolve();REPO=HERE.parents[4];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d16_dedicated_start_specialist";RAW=OUT/"raw"
def mod(name,path):s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d16=mod("d16probe",HERE.parent/"run_phase2_d16_train.py");d3=d16.d3
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
def main():
 p=argparse.ArgumentParser();add_launcher_args(p);args,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0],*hydra];cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=476;cfg.seed=d16.SEED;cfg.episode_length_s=3.;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None
 if args.device:cfg.sim.device=agent.device=args.device
 pool=torch.load(RAW/"validation_start_snapshots.pt",map_location="cpu",weights_only=False);cp=torch.load(RAW/"checkpoints/model_040.pt",map_location="cpu",weights_only=False)
 with launch_simulation(cfg,args):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=d16.StartWorld(wrapped,d3.load_resets(),torch.load(RAW/"train_start_snapshots.pt",map_location="cpu",weights_only=False));policy=d16.StartPolicy(cp["residual_bound"]).to(world.device);policy.residual.load_state_dict(cp["residual_state_dict"]);rows=d16.eval_condition(world,policy,pool,d16.specs()[0]);summary=d16.summarize(rows);summary["residual_l2_mean_active"]=sum(r["residual_l2_mean_active"] for r in rows)/len(rows);summary["residual_max_abs_active"]=max(r["residual_max_abs_active"] for r in rows);d16.dump(RAW/"final_c1_probe.json",{"checkpoint":"model_040.pt","summary":summary,"rows":rows});print(json.dumps(summary,indent=2),flush=True);wrapped.close()
if __name__=="__main__":main()
