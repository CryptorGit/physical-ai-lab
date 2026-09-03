"""Exact 320-environment replay of the D24B pre-START validity boundary."""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import gymnasium as gym
import torch
HERE=Path(__file__).resolve();REPO=HERE.parents[4];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d24c_shold_source_and_safety_parity/raw"
def mod(n,p):s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
b=mod("d24b_exact",HERE.parent/"run_phase2_d24b_transfer.py");d16,d3=b.d16,b.d3
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
def main():
 p=argparse.ArgumentParser();add_launcher_args(p);a,h=setup_preset_cli(p);sys.argv=[sys.argv[0],*h];cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=320;cfg.seed=20279852;cfg.episode_length_s=12.;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None
 if a.device:cfg.sim.device=agent.device=a.device
 pool=torch.load(b.D16/"raw/train_start_snapshots.pt",map_location="cpu",weights_only=False);valid=[i for i,x in enumerate(pool["valid"]) if x][:64]
 with launch_simulation(cfg,a):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=d16.StartWorld(wrapped,d3.load_resets(),pool);hold=d3.initialize("P0_STAND_PARENT",world.device)[0].eval();b.restore(world,pool,valid);obs=world.obs();ok=torch.ones(320,dtype=torch.bool,device=world.device);step_counts=[]
  for _ in range(25):
   with torch.inference_mode():act=hold.mean(obs)
   _,_,done,ex=wrapped.step(act);obs=world.obs();timeout=ex.get("time_outs",torch.zeros_like(done)).bool();step=(world.robot.data.root_lin_vel_b[:,:2].norm(dim=1)<=.08)&(world.robot.data.root_ang_vel_b[:,2].abs()<=.08)&~(done.bool()&~timeout);ok&=step;step_counts.append(int(step.sum()))
  result={"episodes":320,"per_route":64,"continuous_valid":int(ok.sum()),"route_valid":[int(ok[i*64:(i+1)*64].sum()) for i in range(5)],"per_step_valid_counts":step_counts,"exact_D24B_env_count":True,"exact_D24B_seed":True};OUT.mkdir(parents=True,exist_ok=True);(OUT/"exact_d24b_boundary_replay.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2));wrapped.close()
if __name__=="__main__":main()
