"""Run the frozen 50-episode-per-speed steady-state evaluation."""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import gymnasium as gym

SCRIPT=Path(__file__).resolve(); EXP=SCRIPT.parent.parent; REPO=EXP.parents[2]
sys.path.insert(0,str(EXP/"src"))
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
from go2_bidirectional.evaluation import Collector,build_runner,run_steady

p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint",required=True,type=Path);p.add_argument("--output",type=Path,default=REPO/"results/exp_011_unitree_go2_bidirectional_speed_transitions/stage1_single_policy_baseline");p.add_argument("--seed",type=int,default=20260901);add_launcher_args(p);a,h=setup_preset_cli(p);sys.argv=[sys.argv[0],*h]
def main():
 c,agent=resolve_task_config("Isaac-Velocity-Flat-Unitree-Go2-v0","rsl_rl_cfg_entry_point");c.scene.num_envs=50;c.seed=a.seed;c.episode_length_s=10;c.observations.policy.enable_corruption=False;c.events.base_external_force_torque=None;c.events.push_robot=None
 if a.device:c.sim.device=a.device
 a.output.mkdir(parents=True,exist_ok=True)
 with launch_simulation(c,a):
  w,_,policy=build_runner(gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0",cfg=c),agent,a.checkpoint.resolve(strict=True));run_steady(Collector(w,policy),a.output,a.seed);w.close()
if __name__=="__main__":main()

