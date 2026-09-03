"""Phase 2-D5 read-only settle/hold capability formalization runtime."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

HERE=Path(__file__).resolve();EXP=HERE.parent.parent;REPO=EXP.parents[2]
OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d5_settle_hold_capability_contract";RAW=OUT/"raw"
spec=importlib.util.spec_from_file_location("exp014_d3_runtime",HERE.parent/"run_phase2_d3.py");d3=importlib.util.module_from_spec(spec);spec.loader.exec_module(d3)

from g1_explicit_motion_mode.contract import MotionMode, minimum_jerk  # noqa:E402
from g1_explicit_motion_mode.stand_capability_v2 import Exp014ResetToStandEvaluatorV2,Exp014StandHoldEvaluatorV2,legacy_whole_window_2s_average  # noqa:E402
from g1_omnidirectional.policy import FrozenGaitActor  # noqa:E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa:E402
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli  # noqa:E402

DT=.02;SEED=20279051
CANDIDATES={"C0_STAGE2Q":"P1_STOP_PARENT","C1_EXP007_STAND":"P0_STAND_PARENT"}

def dump(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def tensor_hash(*xs):
 h=hashlib.sha256()
 for x in xs:h.update(x.detach().contiguous().cpu().numpy().tobytes())
 return h.hexdigest()
def json_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def rate(x):return float(torch.as_tensor(x,dtype=torch.float32).mean()) if len(x) else 0.

def rollout(world,actor,recipes,capture_boundary=False):
 n=len(recipes);padded=recipes+[recipes[-1]]*(world.env.num_envs-n);obs=world.restore(torch.tensor(padded,device=world.device));local_pos=world.robot.data.root_pos_w[:n]-world.env.scene.env_origins[:n];initial_parts=torch.cat((obs[:n],local_pos,world.robot.data.root_quat_w[:n],world.robot.data.joint_pos[:n],world.robot.data.joint_vel[:n]),1).detach().cpu();initial_hash=tensor_hash(initial_parts)
 fields={k:[] for k in ("speed","yaw","fall","slip","impact","saturation","support")};actions=[];boundary={k:[] for k in ("observation_141","action_37","physical_command","legacy_gait","explicit_mode","previous_mode","previous_command","command_delta","time_since_mode_change","ramp_progress","previous_action")}
 fall=torch.zeros(n,dtype=torch.bool,device=world.device);slip=fall.clone();impact=fall.clone();sat=fall.clone();slip_streak=torch.zeros(n,dtype=torch.long,device=world.device);sat_streak=slip_streak.clone()
 for step in range(200):
  with torch.inference_mode():action=actor.mean(obs)
  actions.append(action[:n].detach().cpu())
  if capture_boundary and step<4:
   boundary["observation_141"].append(obs[:n].detach().cpu());boundary["action_37"].append(action[:n].detach().cpu());boundary["physical_command"].append(world.state.physical_command[:n].detach().cpu());boundary["legacy_gait"].append(torch.zeros(n,1));boundary["explicit_mode"].append(torch.nn.functional.one_hot(world.state.target_mode[:n],3).cpu());boundary["previous_mode"].append(torch.nn.functional.one_hot(world.state.previous_target_mode[:n],3).cpu());boundary["previous_command"].append(world.state.previous_physical_command[:n].detach().cpu());boundary["command_delta"].append((world.state.physical_command-world.state.previous_physical_command)[:n].detach().cpu());boundary["time_since_mode_change"].append(world.state.time_since_mode_change_s[:n].detach().cpu());boundary["ramp_progress"].append(world.state.ramp_progress[:n].detach().cpu());boundary["previous_action"].append(world.env.action_manager.prev_action[:n].detach().cpu())
  obs,_,done,extras=world.step(action,None);timeout=extras.get("time_outs",torch.zeros_like(done)).bool()[:n];fall|=done[:n]&~timeout
  force=world.sensor.data.net_forces_w_history[:n,-1,world.sf,:].norm(dim=-1);contact=force>5;feet=world.robot.data.body_lin_vel_w[:n,world.rf,:2].norm(dim=-1);bad=((feet>.55)&contact).any(1);slip_streak=torch.where(bad,slip_streak+1,torch.zeros_like(slip_streak));slip|=slip_streak>=5;impact|=force.amax(1)>3500;ratio=world.robot.data.joint_vel[:n].abs().div(world.limits[:n].clamp_min(1e-6)).amax(1);sat_streak=torch.where(ratio>.95,sat_streak+1,torch.zeros_like(sat_streak));sat|=sat_streak>=5
  for k,x in (("speed",world.robot.data.root_lin_vel_b[:n,:2].norm(dim=1)),("yaw",world.robot.data.root_ang_vel_b[:n,2].abs()),("fall",fall),("slip",slip),("impact",impact),("saturation",sat),("support",contact.sum(1))):fields[k].append(x.detach().cpu())
 action_tensor=torch.stack(actions);result={k:torch.stack(v) for k,v in fields.items()}|{"recipes":recipes,"initial_state_hash":initial_hash,"action_hash":tensor_hash(action_tensor),"initial_recipe_hashes":[tensor_hash(initial_parts[j]) for j in range(n)],"action_recipe_hashes":[tensor_hash(action_tensor[:,j]) for j in range(n)]}
 if capture_boundary:result["boundary"]={k:torch.stack(v,1) for k,v in boundary.items()}
 return result

def summarize(tr,severity):
 reset_eval=Exp014ResetToStandEvaluatorV2();hold_eval=Exp014StandHoldEvaluatorV2();rows=[]
 edges=torch.quantile(severity,torch.tensor([.25,.5,.75]));reset_pass=[];hold_pass=[];joint=[];legacy=[];acq=[]
 for j,recipe in enumerate(tr["recipes"]):
  kwargs={"fall":tr["fall"][:,j].numpy(),"dangerous_slip":tr["slip"][:,j].numpy(),"impact_failure":tr["impact"][:,j].numpy(),"long_dwell_saturation":tr["saturation"][:,j].numpy()};speed=tr["speed"][:,j].numpy();yaw=tr["yaw"][:,j].numpy();rr=reset_eval.evaluate(speed,yaw,**kwargs);hr=hold_eval.evaluate(speed,yaw,rr,**kwargs);lg=legacy_whole_window_2s_average(speed,yaw,fall=kwargs["fall"][:100],dangerous_slip=kwargs["dangerous_slip"][:100],impact_failure=kwargs["impact_failure"][:100]);reset_pass.append(rr.passed);hold_pass.append(hr.passed if hr.eligible else False);joint.append(rr.passed and bool(hr.passed));legacy.append(lg["passed"]);acq.append(rr.acquisition_time_s if rr.passed else None)
  row={"recipe_id":recipe,"split":d3.split_name(recipe),"severity":float(severity[recipe]),"severity_bin":int(torch.bucketize(severity[recipe],edges)),**{f"reset_{k}":v for k,v in rr.to_dict().items()},**{f"hold_{k}":v for k,v in hr.to_dict().items()},"reset_pass":rr.passed,"hold_pass":hr.passed if hr.eligible else None,"joint_pass":joint[-1],"legacy_pass":lg["passed"],"legacy_speed_mean":lg["speed_mean"],"legacy_absolute_yaw_mean":lg["absolute_yaw_mean"],"reset_speed_trajectory":speed[:100].tolist(),"reset_yaw_trajectory":yaw[:100].tolist()}
  if hr.eligible:row["hold_speed_trajectory"]=speed[hr.start_step:hr.end_step+1].tolist();row["hold_yaw_trajectory"]=yaw[hr.start_step:hr.end_step+1].tolist()
  rows.append(row)
 eligible=[i for i,x in enumerate(reset_pass) if x];at=np.asarray([x for x in acq if x is not None]);n=len(rows)
 reset_safety={k:rate([bool(tr[k][99,j]) for j in range(n)]) for k in ("fall","slip","impact","saturation")}
 hold_safety={k:rate([bool(tr[k][-1,j]) for j in eligible]) for k in ("fall","slip","impact","saturation")}
 summary={"episodes":n,"reset_to_stand_success":sum(reset_pass)/n,"acquisition_time_median":float(np.quantile(at,.5)) if len(at) else None,"acquisition_time_p90":float(np.quantile(at,.9)) if len(at) else None,"acquisition_time_p95":float(np.quantile(at,.95)) if len(at) else None,"hold_after_acquisition_success":sum(reset_pass)/n,"conditional_stand_hold_success":sum(hold_pass)/len(eligible) if eligible else None,"joint_end_to_end_success":sum(joint)/n,"legacy_whole_window_success":sum(legacy)/n,"reset_safety":reset_safety,"hold_safety":hold_safety,"severity_bins":{str(b):{"count":sum(r["severity_bin"]==b for r in rows),"reset_success":rate([r["reset_pass"] for r in rows if r["severity_bin"]==b]),"conditional_hold_success":rate([r["hold_pass"] for r in rows if r["severity_bin"]==b and r["reset_pass"]]),"joint_success":rate([r["joint_pass"] for r in rows if r["severity_bin"]==b])} for b in range(4)}}
 summary["eligible"]=summary["reset_to_stand_success"]>=.95 and all(reset_safety[k]<=({"fall":.02,"slip":.05,"impact":.05,"saturation":.05}[k]) for k in reset_safety) and summary["conditional_stand_hold_success"]>=.95 and all(hold_safety[k]<=({"fall":.02,"slip":.05,"impact":.05,"saturation":.05}[k]) for k in hold_safety) and summary["joint_end_to_end_success"]>=.90
 return summary,rows

def parity_run(world,actor,recipes,severity):
 tr=rollout(world,actor,recipes);summary,rows=summarize(tr,severity);classes=[(r["reset_pass"],r["hold_pass"],r["joint_pass"]) for r in rows];acq=[r["reset_acquisition_time_s"] for r in rows]
 return {"recipe_order":recipes,"recipe_order_hash":json_hash(recipes),"initial_state_hash":tr["initial_state_hash"],"action_hash":tr["action_hash"],"acquisition_times":acq,"acquisition_times_hash":json_hash(acq),"capability_classifications":classes,"classification_hash":json_hash(classes),"aggregate_metrics":summary,"aggregate_metrics_hash":json_hash(summary)}

def paired_same_process_parity(world,actor,severity):
 recipes=d3.VALIDATION;tr=rollout(world,actor,recipes+recipes);runs=[]
 for offset in (0,len(recipes)):
  sub={k:(v[:,offset:offset+len(recipes)] if torch.is_tensor(v) else v) for k,v in tr.items() if k in ("speed","yaw","fall","slip","impact","saturation","support")};sub["recipes"]=recipes;summary,rows=summarize(sub,severity);classes=[(r["reset_pass"],r["hold_pass"],r["joint_pass"]) for r in rows];acq=[r["reset_acquisition_time_s"] for r in rows];runs.append({"recipe_order":recipes,"recipe_order_hash":json_hash(recipes),"initial_recipe_hashes":tr["initial_recipe_hashes"][offset:offset+len(recipes)],"action_recipe_hashes":tr["action_recipe_hashes"][offset:offset+len(recipes)],"acquisition_times":acq,"acquisition_times_hash":json_hash(acq),"capability_classifications":classes,"classification_hash":json_hash(classes),"aggregate_metrics":summary,"aggregate_metrics_hash":json_hash(summary)})
 a,b=runs;comparison={"recipe_order_equal":a["recipe_order"]==b["recipe_order"],"initial_state_hashes_equal":a["initial_recipe_hashes"]==b["initial_recipe_hashes"],"action_hashes_equal":a["action_recipe_hashes"]==b["action_recipe_hashes"],"acquisition_times_equal":a["acquisition_times"]==b["acquisition_times"],"classifications_equal":a["capability_classifications"]==b["capability_classifications"],"aggregate_metrics_equal":a["aggregate_metrics"]==b["aggregate_metrics"],"metric_difference":0 if a["aggregate_metrics"]==b["aggregate_metrics"] else None}
 return {"method":"two simultaneous replicated validation cohorts in one process","runs":runs,"comparison":comparison,"pass":all(v is True or v==0 for v in comparison.values())}

def compare_parity_runs(a,b,method):
 keys=("recipe_order_hash","initial_state_hash","action_hash","acquisition_times_hash","classification_hash","aggregate_metrics_hash")
 comparison={k.replace("_hash","")+"_equal":a[k]==b[k] for k in keys};comparison["metric_difference"]=0 if a["aggregate_metrics"]==b["aggregate_metrics"] else None
 return {"method":method,"runs":[a,b],"comparison":comparison,"pass":all(v is True or v==0 for v in comparison.values())}

def boundary_labels(world,actor,severity):
 parts=[];summaries=[]
 for recipes in (list(range(476)),list(range(476,680))):
  tr=rollout(world,actor,recipes,True);summary,rows=summarize(tr,severity);parts.append((recipes,tr["boundary"]));summaries.extend(rows)
 labels={k:torch.cat([b[k] for _,b in parts],0) for k in parts[0][1]};labels["recipe_id"]=torch.arange(680)[:,None].expand(-1,4);labels["split"]=[d3.split_name(i) for i in range(680)];labels["step"]=torch.arange(4)[None,:].expand(680,-1)
 path=RAW/"Exp014ResetBoundaryLabelAuthorizationV1.pt";path.parent.mkdir(parents=True,exist_ok=True);torch.save(labels,path);finite=all(torch.isfinite(v).all() for v in labels.values() if torch.is_tensor(v) and v.is_floating_point());bounds=bool((labels["action_37"].abs()<=100).all());cont=sum(r["reset_pass"] for r in summaries)/680
 return {"name":"Exp014ResetBoundaryLabelAuthorizationV1","path":str(path.relative_to(REPO)).replace("\\","/"),"sha256":sha(path),"recipes":680,"steps":[0,1,2,3],"samples":2720,"observation_dim":141,"action_dim":37,"missing":0,"nan_inf":0 if finite else 1,"bounds_violation":0 if bounds else 1,"normalized_action_bound":100,"reset_to_stand_physical_continuation":cont,"pre_authorized":bool(finite and bounds and cont>=.95),"added_to_dagger_dataset_v2":False}

def handoff_audit(world,hold_actor):
 n=400;recipes=(d3.VALIDATION[:100]*4);obs=world.restore(torch.tensor(recipes+[recipes[-1]]*(world.env.num_envs-n),device=world.device));dev=world.device;walk=FrozenGaitActor(d3.WMOVE).to(dev).eval();stop=FrozenGaitActor(d3.P1).to(dev).eval();gait=torch.zeros(world.env.num_envs,device=dev);dirs=torch.tensor([0.,90.,180.,270.],device=dev).repeat_interleave(100);rad=torch.deg2rad(dirs);target=torch.zeros(world.env.num_envs,3,device=dev);target[:n,0]=.3*torch.cos(rad);target[:n,1]=.3*torch.sin(rad);world.state.request(torch.full((world.env.num_envs,),int(MotionMode.WALK),device=dev));fall=torch.zeros(n,dtype=torch.bool,device=dev);slip=fall.clone();slip_streak=torch.zeros(n,dtype=torch.long,device=dev)
 def command(x):world.term.external_override[:,:3]=x;world.term._update_command()
 def raw_step(action):
  nonlocal obs,fall,slip,slip_streak
  obs,_,done,extras=world.wrapped.step(action);obs=obs.to(dev);done=done.bool();timeout=extras.get("time_outs",torch.zeros_like(done)).bool()[:n];fall|=done[:n]&~timeout;force=world.sensor.data.net_forces_w_history[:n,-1,world.sf,:].norm(dim=-1);contact=force>5;feet=world.robot.data.body_lin_vel_w[:n,world.rf,:2].norm(dim=-1);bad=((feet>.55)&contact).any(1);slip_streak=torch.where(bad,slip_streak+1,torch.zeros_like(slip_streak));slip|=slip_streak>=5
 for step in range(150):
  p=torch.full((world.env.num_envs,),min(1.,step/75),device=dev);physical=target*minimum_jerk(p)[:,None];world.state.advance(physical,p,DT);command(physical)
  with torch.inference_mode():a=walk(obs["policy"] if isinstance(obs,dict) else world.env.observation_manager.compute()["policy"],gait)
  raw_step(a)
 world.state.request(torch.full((world.env.num_envs,),int(MotionMode.STAND),device=dev));stop_s=[];stop_y=[]
 for step in range(150):
  p=torch.full((world.env.num_envs,),min(1.,step/75),device=dev);physical=target*(1-minimum_jerk(p))[:,None];world.state.advance(physical,p,DT);command(physical);base=world.env.observation_manager.compute()["policy"]
  with torch.inference_mode():a=stop(base,gait)
  raw_step(a)
  if step>=100:stop_s.append(world.robot.data.root_lin_vel_b[:n,:2].norm(dim=1).detach().cpu());stop_y.append(world.robot.data.root_ang_vel_b[:n,2].abs().detach().cpu())
 command(torch.zeros_like(target));base=world.env.observation_manager.compute()["policy"];obs141=world.obs()
 with torch.inference_mode():a_stop=stop(base,gait);a_hold=hold_actor.mean(obs141)
 jump=(a_hold[:n]-a_stop[:n]).norm(dim=1);cos=torch.nn.functional.cosine_similarity(a_hold[:n],a_stop[:n],dim=1);hs=[];hy=[];fall0=fall.clone();slip0=slip.clone()
 for _ in range(100):
  world.state.advance(torch.zeros_like(target),torch.ones(world.env.num_envs,device=dev),DT);command(torch.zeros_like(target));obs141=world.obs()
  with torch.inference_mode():a=hold_actor.mean(obs141)
  raw_step(a);hs.append(world.robot.data.root_lin_vel_b[:n,:2].norm(dim=1).detach().cpu());hy.append(world.robot.data.root_ang_vel_b[:n,2].abs().detach().cpu())
 ss,sy,hs,hy=map(torch.stack,(stop_s,stop_y,hs,hy));rows=[]
 for j in range(n):
  practical=bool(ss[:,j].mean()<=.08 and sy[:,j].mean()<=.08 and not fall0[j] and not slip0[j]);hold_ok=bool(hs[:,j].mean()<=.08 and hy[:,j].mean()<=.08 and torch.quantile(hs[:,j],.95)<=.12 and torch.quantile(hy[:,j],.95)<=.12 and not fall[j] and not slip[j]);rows.append({"direction_deg":float(dirs[j]),"episode":j%100,"recipe_id":recipes[j],"s_stop_practical_stop":practical,"s_hold_2s_hold":hold_ok,"action_jump_l2":float(jump[j]),"action_cosine":float(cos[j]),"fall":bool(fall[j]),"dangerous_slip":bool(slip[j]),"stop_speed_mean":float(ss[:,j].mean()),"stop_absolute_yaw_mean":float(sy[:,j].mean()),"hold_speed_mean":float(hs[:,j].mean()),"hold_speed_p95":float(torch.quantile(hs[:,j],.95)),"hold_absolute_yaw_mean":float(hy[:,j].mean()),"hold_absolute_yaw_p95":float(torch.quantile(hy[:,j],.95))})
 summary={"episodes_per_direction":100,"rows":400,"directions":{str(d):{"s_stop_practical_stop":rate([r["s_stop_practical_stop"] for r in rows if r["direction_deg"]==d]),"s_hold_2s_hold":rate([r["s_hold_2s_hold"] for r in rows if r["direction_deg"]==d]),"fall":rate([r["fall"] for r in rows if r["direction_deg"]==d]),"dangerous_slip":rate([r["dangerous_slip"] for r in rows if r["direction_deg"]==d]),"action_jump_l2_mean":float(np.mean([r["action_jump_l2"] for r in rows if r["direction_deg"]==d])),"action_cosine_mean":float(np.mean([r["action_cosine"] for r in rows if r["direction_deg"]==d]))} for d in (0.,90.,180.,270.)}}
 summary["aggregate"]={"s_stop_practical_stop":rate([r["s_stop_practical_stop"] for r in rows]),"s_hold_2s_hold":rate([r["s_hold_2s_hold"] for r in rows]),"fall":rate([r["fall"] for r in rows]),"dangerous_slip":rate([r["dangerous_slip"] for r in rows]),"action_jump_l2_mean":float(jump.mean()),"action_cosine_mean":float(cos.mean())}
 return summary,rows

def main():
 p=argparse.ArgumentParser();p.add_argument("--mode",choices=("main","fresh-parity","same-parity","same-parity-scenes"),default="main");p.add_argument("--run-id",default="fresh_1");add_launcher_args(p);args,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0],*hydra];RAW.mkdir(parents=True,exist_ok=True);resets=d3.load_resets();severity,_=d3.severity_manifest(resets,json.loads(d3.CFG_PATH.read_text())["severity_weights"])
 cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");standcfg,_=resolve_task_config("Isaac-Velocity-Flat-G1-Run-Stage2-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=476;cfg.seed=20260803;cfg.episode_length_s=20.;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None;cfg.rewards=copy.deepcopy(standcfg.rewards);cfg.terminations=copy.deepcopy(standcfg.terminations)
 if args.device:cfg.sim.device=agent.device=args.device
 with launch_simulation(cfg,args):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=d3.StandWorld(wrapped,resets,severity)
  if args.mode=="fresh-parity":
   selected=json.loads((RAW/"main_results.json").read_text())["selected_candidate"];actor=d3.initialize(CANDIDATES[selected],world.device)[0];dump(RAW/f"parity_{args.run_id}.json",parity_run(world,actor,d3.VALIDATION,severity));wrapped.close();return
  if args.mode=="same-parity":
   selected=json.loads((RAW/"main_results.json").read_text())["selected_candidate"];actor=d3.initialize(CANDIDATES[selected],world.device)[0];dump(RAW/"parity_same_process_paired.json",paired_same_process_parity(world,actor,severity));wrapped.close();return
  if args.mode=="same-parity-scenes":
   selected=json.loads((RAW/"main_results.json").read_text())["selected_candidate"];actor=d3.initialize(CANDIDATES[selected],world.device)[0];run1=parity_run(world,actor,d3.VALIDATION,severity);wrapped.close();wrapped2=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world2=d3.StandWorld(wrapped2,resets,severity);actor2=d3.initialize(CANDIDATES[selected],world2.device)[0];run2=parity_run(world2,actor2,d3.VALIDATION,severity);dump(RAW/"parity_same_process_scenes.json",compare_parity_runs(run1,run2,"two independently constructed scenes in one process"));wrapped2.close();return
  validation={};validation_rows={}
  for name,parent in CANDIDATES.items():
   actor=d3.initialize(parent,world.device)[0];tr=rollout(world,actor,d3.VALIDATION);validation[name],validation_rows[name]=summarize(tr,severity)
  eligible=[n for n in CANDIDATES if validation[n]["eligible"]]
  def key(n):
   x=validation[n];return (-x["joint_end_to_end_success"],-x["reset_to_stand_success"],-x["conditional_stand_hold_success"],x["acquisition_time_p95"],0 if n=="C0_STAGE2Q" else 1)
  selected=sorted(eligible,key=key)[0] if eligible else None;result={"validation":validation,"validation_rows":validation_rows,"eligible":eligible,"selected_candidate":selected,"heldout_opened":False,"aborted_preflight_before_formal_run":True};dump(RAW/"main_results.json",result)
  if selected:
   actor=d3.initialize(CANDIDATES[selected],world.device)[0];held_tr=rollout(world,actor,d3.HELDOUT);held,held_rows=summarize(held_tr,severity);result["heldout_opened"]=True;result["heldout"]={"summary":held,"rows":held_rows,"candidate_frozen_before_open":True,"fallback":False};held_pass=held["reset_to_stand_success"]>=.95 and held["reset_safety"]["fall"]<=.02 and held["reset_safety"]["slip"]<=.05 and held["conditional_stand_hold_success"]>=.95 and held["hold_safety"]["fall"]<=.02 and held["hold_safety"]["slip"]<=.05 and held["joint_end_to_end_success"]>=.90;result["heldout_pass"]=held_pass;dump(RAW/"main_results.json",result)
   if held_pass:
    result["same_process_parity"]=[parity_run(world,actor,d3.VALIDATION,severity),parity_run(world,actor,d3.VALIDATION,severity)];result["boundary"]=boundary_labels(world,actor,severity);dump(RAW/"main_results.json",result);result["handoff"],result["handoff_rows"]=handoff_audit(world,actor)
  dump(RAW/"main_results.json",result);wrapped.close()
 print(json.dumps({"selected":result["selected_candidate"],"eligible":result["eligible"],"heldout_pass":result.get("heldout_pass"),"boundary":result.get("boundary",{}).get("pre_authorized")},indent=2))

if __name__=="__main__":main()
