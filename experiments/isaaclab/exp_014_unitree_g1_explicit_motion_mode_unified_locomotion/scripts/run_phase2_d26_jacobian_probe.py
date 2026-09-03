"""One-step read-only Jacobian API probe for D26 capture diagnostics."""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import gymnasium as gym
import torch

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d3=load("d3_jac_probe",HERE.parent/"run_phase2_d3.py")
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
def main():
    p=argparse.ArgumentParser();add_launcher_args(p);a,h=setup_preset_cli(p);sys.argv=[sys.argv[0],*h]
    cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=1;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None
    out={}
    with launch_simulation(cfg,a):
        w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=d3.StandWorld(w,d3.load_resets(),torch.zeros(680))
        try:
            j=world.robot.root_physx_view.get_jacobians();out["physx"]={"type":str(type(j)),"shape":list(j.shape),"finite":bool(j.isfinite().all()) if hasattr(j,"isfinite") else None}
        except Exception as exc:out["physx_error"]=f"{type(exc).__name__}: {exc}"
        try:
            j=world.robot.root_view.get_jacobians();out["root"]={"type":str(type(j)),"shape":list(j.shape),"finite":bool(j.isfinite().all()) if hasattr(j,"isfinite") else None}
        except Exception as exc:out["root_error"]=f"{type(exc).__name__}: {exc}"
        target=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26_wmove_reference_and_wbik";target.mkdir(parents=True,exist_ok=True);(target/"jacobian_probe.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8");print(json.dumps(out,indent=2),flush=True)
        w.close()
if __name__=="__main__":main()
