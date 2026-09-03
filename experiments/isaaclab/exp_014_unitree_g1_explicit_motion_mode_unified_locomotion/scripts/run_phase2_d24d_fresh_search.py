"""D24D fresh-lifecycle direct-WMOVE control and decisive D23 CEM replay."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, random, sys
from collections import Counter
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch

HERE=Path(__file__).resolve();REPO=HERE.parents[4]
OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d24d_fresh_start_revalidation";RAW=OUT/"raw"
D16=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d16_dedicated_start_specialist";D6=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d6_omnidirectional_stop_teacher";MIRROR=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_d1_yaw_translation_interference_diagnosis/robot_mirror_contract.json"
def mod(n,p):s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d23=mod("d23_d24d",HERE.parent/"run_phase2_d23_lead_reachability.py");d17,d16,d15,d3,d6=d23.d17,d23.d16,d23.d15,d23.d3,d23.d6
from g1_explicit_motion_mode.contract import MotionMode,minimum_jerk
from g1_omnidirectional.policy import FrozenGaitActor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
DT=.02;SEED=20279941
def f(x):return float(x.detach().cpu())
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def mirror_initial(world,mask,contract):
 ids=mask.nonzero().flatten();perm=torch.tensor(contract["mirror_indices"],device=world.device);sign=torch.tensor(contract["mirror_signs"],device=world.device);default=world.robot.data.default_joint_pos[0]
 pos=world.robot.data.root_pos_w[ids].clone();quat=world.robot.data.root_quat_w[ids].clone();vel=torch.cat((world.robot.data.root_lin_vel_w[ids],world.robot.data.root_ang_vel_w[ids]),1).clone();pos[:,1]*=-1;quat[:,1]*=-1;quat[:,3]*=-1;vel[:,1]*=-1;vel[:,3]*=-1;vel[:,5]*=-1
 jp=default+sign*(world.robot.data.joint_pos[ids][:,perm]-default[perm]);jv=sign*world.robot.data.joint_vel[ids][:,perm];world.robot.write_root_pose_to_sim(torch.cat((pos,quat),1),ids);world.robot.write_root_velocity_to_sim(vel,ids);world.robot.write_joint_state_to_sim(jp,jv,env_ids=ids);world.env.sim.forward()
def fresh_lifecycle(world,hold,recipes,mirrored,contract):
 n=len(recipes);pad=recipes+[recipes[-1]]*(world.env.num_envs-n);obs=world.restore(torch.tensor(pad,device=world.device));mm=torch.zeros(world.env.num_envs,dtype=torch.bool,device=world.device);mm[:n]=torch.tensor(mirrored,device=world.device);mirror_initial(world,mm,contract);obs=world.obs();first=torch.full((n,),-1,dtype=torch.long,device=world.device);streak=torch.zeros(n,dtype=torch.long,device=world.device);complete=torch.full_like(first,-1);due=torch.full_like(first,-1);fall=torch.zeros(n,dtype=torch.bool,device=world.device);slip=fall.clone();impact=fall.clone();st=[torch.zeros(n,dtype=torch.long,device=world.device) for _ in range(3)]
 for step in range(240):
  with torch.inference_mode():a=hold.mean(obs)
  obs,_,done,ex=world.step(a,None);sf=d15.safety(world,n,done,ex,st);fall|=sf[0]|sf[5];slip|=sf[1];impact|=sf[2];speed=world.robot.data.root_lin_vel_b[:n,:2].norm(dim=1);yaw=world.robot.data.root_ang_vel_b[:n,2].abs();good=(speed<=.08)&(yaw<=.08);enter=(first<0)&good;first[enter]=step;streak=torch.where(good,streak+1,torch.zeros_like(streak));new=(complete<0)&(streak>=50)&(first<50);complete[new]=step;newdue=(due<0)&(complete>=0)&(step>=complete+50);due[newdue]=step
  if bool((due>=0).all()):break
 valid=(due>=0)&~fall&~slip&~impact
 return obs,valid,{"first":first.detach().cpu().tolist(),"complete":complete.detach().cpu().tolist(),"source_step":due.detach().cpu().tolist(),"fall":fall.detach().cpu().tolist(),"slip":slip.detach().cpu().tolist(),"impact":impact.detach().cpu().tolist()}
def eval_sequence(world,hold,recipes,mirrored,contract,seq,walk,basin,mean,std,thr,lead,trace=False):
 n=len(seq);obs,src_valid,life=fresh_lifecycle(world,hold,recipes,mirrored,contract);source,_=d17.nearest_distance(d17.physical_features(world,n),basin,mean,std);flags=[torch.zeros(n,dtype=torch.bool,device=world.device) for _ in range(6)];st=[torch.zeros(n,dtype=torch.long,device=world.device) for _ in range(3)];first=torch.zeros(n,dtype=torch.bool,device=world.device);first_t=torch.full((n,),-1,dtype=torch.long,device=world.device);acq=torch.zeros(n,dtype=torch.bool,device=world.device);acq_t=torch.full_like(first_t,-1);goodst=torch.zeros(n,dtype=torch.long,device=world.device);flightst=torch.zeros(n,dtype=torch.long,device=world.device);support_loss=torch.zeros(n,dtype=torch.bool,device=world.device);yawmax=torch.zeros(n,device=world.device);smooth=torch.zeros(n,device=world.device);prev=world.env.action_manager.prev_action[:n].clone();tr={k:[] for k in ("vx","vy","yaw","contact","action")} if trace else None
 for step in range(100):
  target=d17.set_command(world,step,.5,n)
  if step<75:a=seq[:,step]
  else:
   with torch.inference_mode():a=walk(world.env.observation_manager.compute()["policy"][:,:123],torch.zeros(world.env.num_envs,device=world.device))[:n]
  if n<world.env.num_envs:a=torch.cat((a,a[-1:].expand(world.env.num_envs-n,-1)),0)
  _,_,done,ex=world.wrapped.step(a);sf=d15.safety(world,n,done,ex,st)[:6]
  for q,z in zip(flags,sf):q|=z
  smooth+=(a[:n]-prev).square().mean(1);prev=a[:n];force=world.sensor.data.net_forces_w_history[:n,-1,world.sf,:].norm(dim=-1);contact=force>5;total=force.sum(1);ratio=total/thr["nominal_force"];flight=(force<5).all(1);flightst=torch.where(flight,flightst+1,torch.zeros_like(flightst));support_loss|=flightst>=5;swing=force[:,0 if lead=="LEFT" else 1]/total.clamp_min(1e-6);support=force[:,1 if lead=="LEFT" else 0]/total.clamp_min(1e-6);vel=world.robot.data.root_lin_vel_b[:n];yaw=world.robot.data.root_ang_vel_b[:n,2];yawmax=torch.maximum(yawmax,yaw.abs());upright=world.robot.data.projected_gravity_b[:n,2]<-.85
  fs=(swing>=thr["swing_low"]-1e-4)&(swing<=thr["swing_high"]+1e-4)&(support>=thr["support_low"]-1e-4)&(support<=thr["support_high"]+1e-4)&(ratio>=thr["total_low"])&(ratio<=thr["total_high"]*1.25)&(yaw.abs()<=.15)&upright&(vel[:,0]>=.03)&(vel[:,0]<=.18);new=(~first)&fs&(step<40);first|=new;first_t[new]=step;good=((vel[:,:2]-target[:,:2]).norm(dim=1)<=.12)&(yaw.abs()<=.10);goodst=torch.where(good,goodst+1,torch.zeros_like(goodst));newa=(~acq)&(goodst>=25)&(step<75);acq|=newa;acq_t[newa]=step
  if trace:
   tr["vx"].append(vel[:,0].detach().cpu());tr["vy"].append(vel[:,1].detach().cpu());tr["yaw"].append(yaw.detach().cpu());tr["contact"].append(contact.detach().cpu());tr["action"].append(a[:n].detach().cpu())
 final,_=d17.nearest_distance(d17.physical_features(world,n),basin,mean,std);hard=flags[0]|flags[1]|flags[2]|flags[3]|flags[4]|flags[5]|support_loss|~src_valid;safe=~hard;out={"source_valid":src_valid,"safe":safe,"first":first&safe,"acq":acq&safe,"first_t":first_t,"acq_t":acq_t,"basin_ratio":final/source.clamp_min(1e-6),"yaw":yawmax,"smooth":smooth/100,"flags":flags,"support_loss":support_loss,"lifecycle":life};out["trace"]={k:torch.stack(v).numpy().tolist() for k,v in tr.items()} if trace else None;return out
def fresh_cem(world,hold,recipe,mirrored,contract,lead,hold_action,move,start_basis,pca,walk,basin,mean,std,thr,seed):
 torch.manual_seed(seed);mu=torch.zeros(1,15,12,device=world.device);sd=torch.full_like(mu,.35);hist=[];global_score=-float("inf");global_coef=None;reasons=Counter()
 for it in range(12):
  coef=(mu+sd*torch.randn(128,15,12,device=world.device)).clamp(-1,1);coef[0]=mu[0];seq=d23.sequence(coef,hold_action.expand(128,-1),move.expand(128,-1),start_basis,pca);out=eval_sequence(world,hold,[recipe]*128,[mirrored]*128,contract,seq,walk,basin,mean,std,thr,lead);viol=sum(z.float() for z in out["flags"])+out["support_loss"].float()+(~out["source_valid"]).float();score=torch.where(out["safe"],torch.zeros_like(out["basin_ratio"]),-1e9-viol);score+=out["first"].float()*1e6+out["acq"].float()*1e4-out["basin_ratio"]*100-out["yaw"]-out["smooth"]*.1;elite=score.topk(13).indices;chosen=coef[elite];mu=chosen.mean(0,keepdim=True);sd=chosen.std(0,keepdim=True).clamp(.03,.5);bi=score.argmax();bs=f(score[bi])
  if bs>global_score:global_score=bs;global_coef=coef[bi:bi+1].detach().clone()
  labels=("fall","dangerous_slip","impact","velocity_saturation","torque_saturation","nonfinite")
  for name,z in zip(labels,out["flags"]):reasons[name]+=int(z.sum())
  reasons["support_loss"]+=int(out["support_loss"].sum());reasons["source_invalid"]+=int((~out["source_valid"]).sum());hist.append({"iteration":it,"safe_candidates":int(out["safe"].sum()),"first_step_candidates":int(out["first"].sum()),"walk_acquisition_candidates":int(out["acq"].sum()),"best_score":bs})
 best=d23.sequence(global_coef,hold_action,move,start_basis,pca);final=eval_sequence(world,hold,[recipe],[mirrored],contract,best,walk,basin,mean,std,thr,lead,True);fl=final["flags"]
 return best.detach().cpu(),{"recipe_id":recipe,"source_kind":"MIRRORED" if mirrored else "ORIGINAL","lead":lead,"safe_global_best":bool(final["safe"][0]),"first_step":bool(final["first"][0]),"walk_acquisition":bool(final["acq"][0]),"confirmation":bool(final["acq"][0]),"first_step_time_step":int(final["first_t"][0]),"acquisition_time_step":int(final["acq_t"][0]),"basin_ratio":f(final["basin_ratio"][0]),"yaw_p95_proxy":f(final["yaw"][0]),"fall":bool(fl[0][0]),"dangerous_slip":bool(fl[1][0]),"impact":bool(fl[2][0]),"velocity_saturation":bool(fl[3][0]),"torque_saturation":bool(fl[4][0]),"nonfinite":bool(fl[5][0]),"support_loss":bool(final["support_loss"][0]),"candidate_evaluations":1536,"verification_evaluations":1,"hard_reject_reason_distribution":dict(reasons),"fitness_evolution":hist,"trace":final["trace"]}
def direct_control(world,hold,walk,contract):
 rows=[]
 for recipe,mirrored in [(i,False) for i in range(4)]+[(i,True) for i in range(4)]:
  obs,valid,life=fresh_lifecycle(world,hold,[recipe],[mirrored],contract);root0=world.robot.data.root_pos_w[0,0].clone();first=-1;complete=-1;streak=0;flags=[False]*6;st=[torch.zeros(1,dtype=torch.long,device=world.device) for _ in range(3)];yaws=[];contacts=[]
  world.state.request(torch.full((world.env.num_envs,),int(MotionMode.WALK),device=world.device))
  for step in range(100):
   target=d17.set_command(world,step,.5,1);o=world.env.observation_manager.compute()["policy"]
   with torch.inference_mode():a=walk(o[:,:123],torch.zeros(world.env.num_envs,device=world.device))
   _,_,done,ex=world.wrapped.step(a);sf=d15.safety(world,1,done,ex,st)[:6];flags=[q or bool(z[0]) for q,z in zip(flags,sf)];force=world.sensor.data.net_forces_w_history[0,-1,world.sf,:].norm(dim=-1);contact=force>5;contacts.append(contact.cpu().tolist());yaw=f(world.robot.data.root_ang_vel_b[0,2]);yaws.append(yaw);vx=f(world.robot.data.root_lin_vel_b[0,0]);vy=f(world.robot.data.root_lin_vel_b[0,1]);
   if first<0 and int(contact.sum())==1 and f(world.robot.data.root_pos_w[0,0]-root0)>0:first=step
   good=((vx-.3)**2+vy**2)**.5<=.12 and abs(yaw)<=.10;streak=streak+1 if good else 0
   if complete<0 and streak>=25 and step<75:complete=step
  signs=sum(1 for a,b in zip(yaws,yaws[1:]) if a*b<0);rows.append({"recipe_id":recipe,"source_kind":"MIRRORED" if mirrored else "ORIGINAL","source_valid":bool(valid[0]),"first_step":first>=0,"first_step_step":first,"walk_acquisition":complete>=0,"confirmation":complete>=0,"confirmation_step":complete,"yaw_p95":float(np.quantile(np.abs(yaws),.95)),"yaw_sign_changes":signs,"contact_sequence":contacts,"fall":flags[0],"dangerous_slip":flags[1],"impact":flags[2],"velocity_saturation":flags[3],"torque_saturation":flags[4],"nonfinite":flags[5],"lifecycle":life})
 return rows
def main():
 p=argparse.ArgumentParser();add_launcher_args(p);a,h=setup_preset_cli(p);sys.argv=[sys.argv[0],*h];cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=128;cfg.seed=SEED;cfg.episode_length_s=20.;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None
 if a.device:cfg.sim.device=agent.device=a.device
 torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);train=torch.load(D16/"raw/train_start_snapshots.pt",map_location="cpu",weights_only=False);d6pool=torch.load(D6/"raw/snapshots/selected/train_batch_00.pt",map_location="cpu",weights_only=False);contract=json.loads(MIRROR.read_text());old=json.loads((REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d23_explicit_lead_start_reachability/raw/worker_results.json").read_text())
 with launch_simulation(cfg,a):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=d16.StartWorld(wrapped,d3.load_resets(),train);walk=FrozenGaitActor(d3.WMOVE).to(world.device).eval();hold=d3.initialize("P0_STAND_PARENT",world.device)[0].eval();basin,_,by_ref,thr=d23.collect_reference(world,d6pool,walk);basin=basin.to(world.device);by_ref=by_ref.to(world.device);mean,std=basin.mean(0),basin.std(0).clamp_min(1e-4);names=contract["joint_names"];left=d23.action_modes(names,contract,"LEFT").to(world.device);right=d23.action_modes(names,contract,"RIGHT").to(world.device);center=by_ref-by_ref.mean(0);_,_,v=torch.pca_lowrank(center,q=8,center=False);pca=v[:,:8].T.contiguous();direct=direct_control(world,hold,walk,contract);results=[];seqs=[];source_manifest=[]
  for ix,(recipe,mirrored) in enumerate([(i,False) for i in range(4)]+[(i,True) for i in range(4)]):
   obs,valid,life=fresh_lifecycle(world,hold,[recipe],[mirrored],contract);source=d17.physical_features(world,1);_,near=d17.nearest_distance(source,basin,mean,std);move=by_ref[near];ha=hold.mean(obs[:1]).detach();source_manifest.append({"development_index":ix,"recipe_id":recipe,"kind":"MIRRORED" if mirrored else "ORIGINAL","mirror_pair":ix-4 if mirrored else ix+4,"fresh_source_valid":bool(valid[0]),"lifecycle":life,"observation_hash":hashlib.sha256(obs[0].detach().cpu().numpy().tobytes()).hexdigest()})
   for li,lead in enumerate(("LEFT","RIGHT")):
    seq,r=fresh_cem(world,hold,recipe,mirrored,contract,lead,ha,move,left if lead=="LEFT" else right,pca,walk,basin,mean,std,thr,20279711+ix*2+li);r["development_index"]=ix;results.append(r);seqs.append(seq.numpy())
  RAW.mkdir(parents=True,exist_ok=True);np.savez_compressed(RAW/"fresh_best_sequences.npz",**{f"seq_{i:02d}":x for i,x in enumerate(seqs)});dump(RAW/"fresh_search_results.json",{"seed":SEED,"source_manifest":source_manifest,"thresholds":thr,"basis":{"left":left.cpu().tolist(),"right":right.cpu().tolist(),"pca_components":8},"direct_control":direct,"search":results,"old_D23":old["development"],"candidate_evaluations":sum(r["candidate_evaluations"] for r in results),"verification_evaluations":16,"persistent_updates":0,"validation_access":0,"heldout_access":0});print(json.dumps({"direct_acquisition":sum(r["walk_acquisition"] for r in direct),"first":sum(any(r["development_index"]==i and r["first_step"] for r in results) for i in range(8)),"acq":sum(any(r["development_index"]==i and r["walk_acquisition"] for r in results) for i in range(8)),"candidate_evaluations":sum(r["candidate_evaluations"] for r in results)},indent=2));wrapped.close()
if __name__=="__main__":main()
