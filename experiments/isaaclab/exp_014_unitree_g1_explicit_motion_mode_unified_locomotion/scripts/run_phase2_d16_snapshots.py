"""Generate result-blind S_HOLD start-state pools for D16."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import gymnasium as gym
import torch

HERE=Path(__file__).resolve();REPO=HERE.parents[4];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d16_dedicated_start_specialist";RAW=OUT/"raw"
def mod(name,path):s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d15=mod("d15snap",HERE.parent/"run_phase2_d15_worker.py");d3=d15.d3
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
def tensor_hash(x):
 h=hashlib.sha256()
 for k in sorted(x):h.update(x[k].contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--split",choices=("train","validation"),required=True);add_launcher_args(p);args,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0],*hydra]
 recipes=d3.TRAIN if args.split=="train" else d3.VALIDATION;n=len(recipes);cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=n if args.split=="train" else 128;cfg.seed=20279301 if args.split=="train" else 20279215;cfg.episode_length_s=20.;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None
 if args.device:cfg.sim.device=agent.device=args.device
 with launch_simulation(cfg,args):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=d3.StandWorld(wrapped,d3.load_resets(),torch.zeros(680));hold=d3.initialize("P0_STAND_PARENT",world.device)[0].eval();x=d15.stand_snapshots(world,hold,recipes)
  x["recipes"]=recipes;x["split"]=args.split;x["snapshot_tensor_hash"]=tensor_hash(x["snapshot"]);RAW.mkdir(parents=True,exist_ok=True);torch.save(x,RAW/f"{args.split}_start_snapshots.pt")
  manifest={k:v for k,v in x.items() if k not in ("snapshot","pre_actions")};manifest["count"]=n;manifest["valid_count"]=sum(x["valid"]);manifest["invalid_recipe_ids"]=[r for r,v in zip(recipes,x["valid"]) if not v];(RAW/f"{args.split}_start_snapshot_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
  print(json.dumps({"split":args.split,"count":n,"valid":sum(x["valid"]),"tensor_hash":x["snapshot_tensor_hash"]},indent=2),flush=True);wrapped.close()
if __name__=="__main__":main()
