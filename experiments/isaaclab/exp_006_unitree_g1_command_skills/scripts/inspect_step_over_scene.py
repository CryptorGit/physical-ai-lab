"""Inspect instantiated foot collision keypoints and obstacle geometry."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import gymnasium as gym,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[2];sys.path[:0]=[str(ROOT/"src"),str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src")]
import isaaclab_tasks,g1_flat_run.tasks,g1_command_skills.tasks  # noqa:E402,F401
from isaaclab.utils.math import quat_apply  # noqa:E402
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli  # noqa:E402
p=argparse.ArgumentParser();p.add_argument("--output",required=True);add_launcher_args(p);a,h=setup_preset_cli(p);sys.argv=[sys.argv[0]]+h
def main():
 cfg,_=resolve_task_config("Isaac-Motion-Flat-G1-Command-StepOverAudit-Eval-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=1
 with launch_simulation(cfg,a):
  raw=gym.make("Isaac-Motion-Flat-G1-Command-StepOverAudit-Eval-v0",cfg=cfg);raw.reset();e=raw.unwrapped;r=e.scene["robot"];ids,names=r.find_bodies(["left_ankle_roll_link","right_ankle_roll_link"],preserve_order=True);pos=r.data.body_pos_w.torch[:,ids];quat=r.data.body_quat_w.torch[:,ids]
  keys={"toe_bottom":[.06383880963290349,0,-.025807180037281774],"sole_center":[.04321213238651294,0,-.025807180037281774],"heel_bottom":[.022585455140122387,0,-.025807180037281774]};world={}
  for k,v in keys.items():world[k]=(pos+quat_apply(quat.reshape(-1,4),torch.tensor(v,device=e.device).expand(1,2,3).reshape(-1,3)).reshape(1,2,3))[0].tolist()
  result={"source_usd":str(cfg.scene.robot.spawn.usd_path),"foot_bodies":names,"keypoints_body_m":keys,"keypoints_world_at_reset_m":world,"obstacle":{"center_x_m":.32,"front_x_m":.29,"rear_x_m":.35,"height_m":.05,"depth_m":.06,"width_m":2.2},"lead_landing_target_x_m":.41}
  out=Path(a.output).resolve();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2));raw.close()
if __name__=="__main__":main()
