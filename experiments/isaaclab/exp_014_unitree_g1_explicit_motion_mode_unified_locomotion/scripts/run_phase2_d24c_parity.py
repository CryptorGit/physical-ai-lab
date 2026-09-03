"""D24C train-only S_HOLD lifecycle/restore/saturation parity runtime."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d24c_shold_source_and_safety_parity"; RAW=OUT/"raw"
D16=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d16_dedicated_start_specialist/raw/train_start_snapshots.pt"
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
d5=loadmod("d5_d24c",HERE.parent/"run_phase2_d5.py"); d3=d5.d3
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli

DT=.02; SEED=20279901
def dump(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def th(x): return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
def mh(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()): h.update(k.encode()); h.update(v.detach().contiguous().cpu().numpy().tobytes())
 return h.hexdigest()
def tensor_info(world,n):
 s=world.sensor.data; r=world.robot.data
 hist=getattr(s,"net_forces_w_history",None); air=getattr(s,"current_air_time",None); last=getattr(s,"last_air_time",None)
 vals={"root":r.root_state_w[:n],"joint_pos":r.joint_pos[:n],"joint_vel":r.joint_vel[:n],"previous_action":world.env.action_manager.prev_action[:n],"action":world.env.action_manager.action[:n],"obs_141":world.obs()[:n],"contact_force":hist[:n] if hist is not None else None,"air_time":air[:n] if air is not None else None,"last_air_time":last[:n] if last is not None else None,"computed_torque":getattr(r,"computed_torque",None),"applied_torque":r.applied_torque[:n],"effort_limit":r.joint_effort_limits[:n]}
 return {k:None if v is None else {"hash":th(v[:n] if v.ndim and v.shape[0]>=n else v),"shape":list(v.shape),"finite":bool(torch.isfinite(v).all())} for k,v in vals.items()}
def run_restored(world,hold,snapshot,n,warmup):
 snap={k:v[:world.env.num_envs].to(world.device) for k,v in snapshot.items()}; obs=world.restore_snapshot(snap); immediate=tensor_info(world,n)
 for _ in range(warmup):
  with torch.inference_mode(): a=hold.mean(obs)
  obs,_,_,_=world.step(a,None)
 one=tensor_info(world,n) if warmup==1 else None
 valid=torch.ones(n,dtype=torch.bool,device=world.device); fall=torch.zeros_like(valid); slip=fall.clone(); impact=fall.clone(); cvsat=fall.clone(); dtorque=fall.clone(); ss=torch.zeros(n,dtype=torch.long,device=world.device); vs=ss.clone(); ts=ss.clone(); contact_match=[]; speeds=[]; yaws=[]
 for _ in range(25):
  with torch.inference_mode(): a=hold.mean(obs)
  obs,_,done,ex=world.step(a,None); timeout=ex.get("time_outs",torch.zeros_like(done)).bool()[:n]; fall|=done[:n]&~timeout
  force=world.sensor.data.net_forces_w_history[:n,-1,world.sf,:].norm(dim=-1); c=force>5; fv=world.robot.data.body_lin_vel_w[:n,world.rf,:2].norm(dim=-1); bad=((fv>.55)&c).any(1); ss=torch.where(bad,ss+1,torch.zeros_like(ss)); slip|=ss>=5; impact|=force.amax(1)>3500
  vr=world.robot.data.joint_vel[:n].abs().div(world.limits[:n].clamp_min(1e-6)).amax(1); vs=torch.where(vr>.95,vs+1,torch.zeros_like(vs)); cvsat|=vs>=5
  eff=world.robot.data.joint_effort_limits[:n].abs().clamp_min(1e-6); tr=world.robot.data.applied_torque[:n].abs().div(eff).amax(1); ts=torch.where(tr>.95,ts+1,torch.zeros_like(ts)); dtorque|=ts>5
  speed=world.robot.data.root_lin_vel_b[:n,:2].norm(dim=1); yaw=world.robot.data.root_ang_vel_b[:n,2].abs(); valid&=(speed<=.08)&(yaw<=.08)&~fall&~slip&~impact; speeds.append(speed.detach().cpu()); yaws.append(yaw.detach().cpu()); contact_match.append(c.sum(1).detach().cpu())
 return {"episodes":n,"warmup_steps":warmup,"d24b_continuous_validity":float(valid.float().mean()),"fall":float(fall.float().mean()),"dangerous_slip":float(slip.float().mean()),"impact":float(impact.float().mean()),"canonical_velocity_saturation":float(cvsat.float().mean()),"d24b_torque_long_dwell":float(dtorque.float().mean()),"speed_mean":float(torch.stack(speeds).mean()),"yaw_mean":float(torch.stack(yaws).mean()),"double_support_fraction":float((torch.stack(contact_match)==2).float().mean()),"immediate_identity":immediate,"one_step_identity":one}
def main():
 p=argparse.ArgumentParser(); add_launcher_args(p); a,h=setup_preset_cli(p); sys.argv=[sys.argv[0],*h]
 cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point"); cfg.scene.num_envs=100; cfg.seed=SEED; cfg.episode_length_s=20.; cfg.observations.policy.enable_corruption=False; cfg.events.base_external_force_torque=None; cfg.events.push_robot=None
 if a.device: cfg.sim.device=agent.device=a.device
 RAW.mkdir(parents=True,exist_ok=True)
 resets=d3.load_resets(); severity,_=d3.severity_manifest(resets,json.loads(d3.CFG_PATH.read_text())["severity_weights"]); pool=torch.load(D16,map_location="cpu",weights_only=False); valid=[i for i,x in enumerate(pool["valid"]) if x][:64]; old={k:torch.cat((v[valid],v[valid[-1]:valid[-1]+1].repeat((36,)+(1,)*(v.ndim-1))),0) for k,v in pool["snapshot"].items()}
 with launch_simulation(cfg,a):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions); world=d3.StandWorld(wrapped,resets,severity); hold=d3.initialize("P0_STAND_PARENT",world.device)[0].eval()
  fresh=d5.rollout(world,hold,d3.TRAIN[:100]); fresh_summary,fresh_rows=d5.summarize(fresh,severity); torch.save(fresh,RAW/"fresh_shold_trace.pt")
  # M0 is the final 25 steps of the uninterrupted canonical lifecycle.
  m0valid=((fresh["speed"][-25:]<=.08)&(fresh["yaw"][-25:]<=.08)&~fresh["fall"][-25:]&~fresh["slip"][-25:]&~fresh["impact"][-25:]).all(0)
  modes={"M0_FRESH_LIFECYCLE":{"episodes":100,"canonical_joint_success":fresh_summary["joint_end_to_end_success"],"d24b_continuous_validity":float(m0valid.float().mean()),"canonical_velocity_saturation":fresh_summary["reset_safety"]["saturation"],"d24b_torque_long_dwell":"NOT_EVALUATED_IN_CANONICAL_TRACE"}}
  modes["M1_D21_SNAPSHOT_RESTORE"]=run_restored(world,hold,old,64,0)
  modes["M2_FULL_STATE_RESTORE"]={"status":"NOT_AVAILABLE","reason":"committed snapshot has no actuator, contact-history, air-time, termination, or observation-history tensors"}
  for w in (1,2,4,8,16): modes[f"M3_RESTORE_WARMUP_{w}"]=run_restored(world,hold,old,64,w)
  dump(RAW/"parity_results.json",{"fresh_summary":fresh_summary,"fresh_rows":fresh_rows,"modes":modes,"snapshot_pool_indices":valid,"snapshot_keys":sorted(pool["snapshot"]),"snapshot_valid_count":sum(pool["valid"]),"teacher_tensor_hash":mh(hold),"persistent_update":0,"validation_access":0,"heldout_access":0})
  print(json.dumps({"fresh":fresh_summary,"modes":{k:{kk:vv for kk,vv in v.items() if kk not in ("immediate_identity","one_step_identity")} for k,v in modes.items()}},indent=2)); wrapped.close()
if __name__=="__main__": main()
