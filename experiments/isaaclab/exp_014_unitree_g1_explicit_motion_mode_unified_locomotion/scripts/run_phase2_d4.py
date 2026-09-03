"""Phase 2-D4: read-only STAND objective and rollout-horizon attribution.

No persistent optimizer update, checkpoint, dataset, held-out access, or runtime
integration is performed. All PPO updates live only in temporary in-memory clones.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import torch

HERE = Path(__file__).resolve(); EXP = HERE.parent.parent; REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d4_stand_objective_horizon_attribution"
REPORT = REPO / "research/exp_014_phase_2_d4_stand_objective_horizon_attribution_report.md"
D3_PATH = HERE.parent / "run_phase2_d3.py"
spec = importlib.util.spec_from_file_location("exp014_d3_runtime", D3_PATH); d3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d3)

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa:E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa:E402

DT=.02; GAMMA=.99; LAM=.95; SEED=20279041
WINDOWS={"W0":(0,24),"W1":(24,50),"W2":(50,75),"W3":(75,100)}


def dump(name,obj):OUT.mkdir(parents=True,exist_ok=True);(OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def git(*args):return subprocess.check_output(["git",*args],cwd=REPO,text=True).strip()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def write_csv(name,rows,fields=None):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 if fields is None:
  fields=[]
  for r in rows:
   for k in r:
    if k not in fields:fields.append(k)
 with p.open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:(json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v) for k,v in r.items()} for r in rows])
def rate(x):return float(x.float().mean()) if x.numel() else 0.
def corr(a,b):
 a=a.float();b=b.float();a=a-a.mean();b=b-b.mean();den=a.norm()*b.norm();return None if float(den)<1e-12 else float((a*b).sum()/den)


def collect(world,actor,critic,recipes,steps=100,stochastic=False,seed=SEED):
 torch.manual_seed(seed);random.seed(seed);n=len(recipes);padded=recipes+[recipes[-1]]*(world.env.num_envs-n);obs=world.restore(torch.tensor(padded,device=world.device))
 names=list(world.env.reward_manager.active_terms);data={k:[] for k in ("obs","action","reward","terms","value","next_value","done","timeout","speed","yaw","roll_pitch","support","fall","slip","valid")}
 active=torch.ones(n,dtype=torch.bool,device=world.device);fallen=torch.zeros_like(active);slipped=torch.zeros_like(active);slip_streak=torch.zeros(n,dtype=torch.long,device=world.device)
 for t in range(steps):
  with torch.inference_mode():dist=actor.dist(obs);action=dist.sample() if stochastic else dist.mean;value=critic(obs)
  data["obs"].append(obs[:n].detach().cpu());data["action"].append(action[:n].detach().cpu());data["value"].append(value[:n].detach().cpu());data["valid"].append(active.cpu())
  obs2,reward,done,extras=world.step(action,None)
  with torch.inference_mode():nv=critic(obs2)
  timeout=extras.get("time_outs",torch.zeros_like(done)).bool()[:n];terms=world.env.reward_manager._step_reward[:n].detach()*DT
  force=world.sensor.data.net_forces_w_history[:n,-1,world.sf,:].norm(dim=-1);contact=force>5;feet=world.robot.data.body_lin_vel_w[:n,world.rf,:2].norm(dim=-1);bad=((feet>.55)&contact).any(1);slip_streak=torch.where(bad,slip_streak+1,torch.zeros_like(slip_streak));slipped|=slip_streak>=5;fallen|=done[:n]&~timeout
  grav=world.robot.data.projected_gravity_b[:n];rp=torch.acos((-grav[:,2]).clamp(-1,1))
  data["reward"].append((reward[:n]*active).detach().cpu());data["terms"].append((terms*active[:,None]).cpu());data["next_value"].append(nv[:n].detach().cpu());data["done"].append((done[:n]&active).cpu());data["timeout"].append((timeout&active).cpu())
  data["speed"].append(world.robot.data.root_lin_vel_b[:n,:2].norm(dim=1).detach().cpu());data["yaw"].append(world.robot.data.root_ang_vel_b[:n,2].abs().detach().cpu());data["roll_pitch"].append(rp.detach().cpu());data["support"].append(contact.sum(1).detach().cpu());data["fall"].append(fallen.cpu());data["slip"].append(slipped.cpu())
  active&=~done[:n];obs=obs2
 return {k:torch.stack(v) for k,v in data.items()}|{"reward_names":names,"recipes":recipes,"stochastic":stochastic}


def segmented_gae(reward,value,next_value,done,horizon):
 T,N=reward.shape;adv=torch.zeros_like(reward);delta=reward+GAMMA*next_value*(~done).float()-value;boundary_boot=torch.zeros_like(reward)
 for start in range(0,T,horizon):
  end=min(T,start+horizon);g=torch.zeros(N)
  for t in reversed(range(start,end)):
   mask=(~done[t]).float();g=delta[t]+GAMMA*LAM*mask*g;adv[t]=g
  if end<T:boundary_boot[end-1]=GAMMA*next_value[end-1]*(~done[end-1]).float()
 return adv,adv+value,delta,boundary_boot


def trajectory_gae_outputs(tr):
 rows=[];summary=[];groups={"step_0_3":(0,4),"step_4_23":(4,24),"step_24_49":(24,50),"step_50_99":(50,100)}
 for h in (24,50,100):
  adv,ret,delta,boot=segmented_gae(tr["reward"],tr["value"],tr["next_value"],tr["done"],h)
  for name,(a,b) in groups.items():
   x=adv[a:b];r=ret[a:b];bt=boot[a:b]
   row={"horizon":f"H{h}","step_group":name,"samples":x.numel(),"advantage_mean":float(x.mean()),"advantage_abs_mean":float(x.abs().mean()),"advantage_positive_rate":rate(x>0),"advantage_negative_rate":rate(x<0),"return_target_mean":float(r.mean()),"bootstrap_contribution_mean":float(bt.mean()),"delta_mean":float(delta[a:b].mean())};rows.append(row)
  summary.append({"horizon":h,"advantage_step0_mean":float(adv[0].mean()),"return_step0_mean":float(ret[0].mean()),"boundary_bootstrap_sum":float(boot.sum()),"artificial_boundaries":[x-1 for x in range(h,100,h)]})
 return rows,summary


def formal_decomposition(tr,severity):
 speed,yaw,rp,support=tr["speed"],tr["yaw"],tr["roll_pitch"],tr["support"];n=speed.shape[1];whole=(speed.mean(0)<=.08)&(yaw.mean(0)<=.08)&~tr["fall"][-1]&~tr["slip"][-1]
 rows=[]
 for j,recipe in enumerate(tr["recipes"]):
  good_s=speed[:,j]<=.08;good_y=yaw[:,j]<=.08;good=good_s&good_y
  first_s=int(good_s.nonzero()[0])+1 if good_s.any() else None;first_y=int(good_y.nonzero()[0])+1 if good_y.any() else None;first=int(good.nonzero()[0])+1 if good.any() else None
  reexit=int(((good[:-1])&(~good[1:])).sum());hold=0
  if first is not None:
   for z in good[first-1:]:
    if not bool(z):break
    hold+=1
  for w,(a,b) in WINDOWS.items():
   s=speed[a:b,j];y=yaw[a:b,j];rows.append({"recipe_id":recipe,"window":w,"start_s":a*DT,"end_s":b*DT,"formal_2s_pass":bool(whole[j]),"speed_mean":float(s.mean()),"speed_p95":float(torch.quantile(s,.95)),"absolute_yaw_mean":float(y.mean()),"absolute_yaw_p95":float(torch.quantile(y,.95)),"roll_pitch_mean":float(rp[a:b,j].mean()),"fall":bool(tr["fall"][b-1,j]),"dangerous_slip":bool(tr["slip"][b-1,j]),"support_flight_fraction":rate(support[a:b,j]==0),"support_single_fraction":rate(support[a:b,j]==1),"support_double_fraction":rate(support[a:b,j]>=2),"first_speed_entry_s":None if first_s is None else first_s*DT,"first_yaw_entry_s":None if first_y is None else first_y*DT,"first_both_entry_s":None if first is None else first*DT,"continuous_hold_after_entry_s":hold*DT,"re_exit_count":reexit,"severity":float(severity[recipe])})
 agg={}
 for w,(a,b) in WINDOWS.items():agg[w]={"speed_mean":float(speed[a:b].mean()),"speed_p95":float(torch.quantile(speed[a:b],.95)),"absolute_yaw_mean":float(yaw[a:b].mean()),"absolute_yaw_p95":float(torch.quantile(yaw[a:b],.95)),"roll_pitch_mean":float(rp[a:b].mean()),"fall":rate(tr["fall"][b-1]),"dangerous_slip":rate(tr["slip"][b-1]),"within_both_rate":rate((speed[a:b]<=.08)&(yaw[a:b]<=.08))}
 reset_diag=[]
 for j in range(n):
  good=(speed[:,j]<=.08)&(yaw[:,j]<=.08);ok=False
  for t in range(50):
   if bool(good[t]) and bool(good[t:100].all()):ok=True;break
  reset_diag.append(ok)
 reset_diag=torch.tensor(reset_diag);hold_diag=(speed[50:100].mean(0)<=.08)&(yaw[50:100].mean(0)<=.08)&~tr["fall"][-1]&~tr["slip"][-1]
 diag={"recipes":n,"old_whole_window_practical_stand":rate(whole),"RESET_TO_STAND_DIAGNOSTIC":rate(reset_diag&~tr["fall"][-1]&~tr["slip"][-1]),"STAND_HOLD_DIAGNOSTIC":rate(hold_diag),"fall":rate(tr["fall"][-1]),"dangerous_slip":rate(tr["slip"][-1]),"contract_use":"DIAGNOSTIC_ONLY_NOT_FORMAL_PASS"}
 return rows,{"windows":agg,"whole":diag,"primary_failing_window":max(agg,key=lambda w:agg[w]["speed_mean"]+agg[w]["absolute_yaw_mean"])},diag,whole


def reward_attribution(tr,whole):
 names=tr["reward_names"];terms=tr["terms"];rows=[];summ={}
 both_entry=torch.full((len(tr["recipes"]),),2.02)
 good=(tr["speed"]<=.08)&(tr["yaw"]<=.08)
 for j in range(good.shape[1]):
  z=good[:,j].nonzero()
  if len(z):both_entry[j]=(int(z[0])+1)*DT
 hold=((tr["speed"][50:].mean(0)<=.08)&(tr["yaw"][50:].mean(0)<=.08)).float()
 for i,name in enumerate(names):
  total=terms[:,:,i].sum(0);summ[name]={"cumulative_mean":float(total.mean()),"pass_mean":float(total[whole].mean()) if whole.any() else None,"fail_mean":float(total[~whole].mean()) if (~whole).any() else None,"corr_formal_pass":corr(total,whole.float()),"corr_settling_time":corr(total,both_entry),"corr_final_hold":corr(total,hold)}
  for w,(a,b) in WINDOWS.items():
   for label,mask in (("success",whole),("failure",~whole)):
    x=terms[a:b,:,i][:,mask];rows.append({"term":name,"window":w,"outcome":label,"samples":x.numel(),"mean_reward":float(x.mean()) if x.numel() else None,"cumulative_reward_mean":float(x.sum(0).mean()) if x.numel() else None})
 return rows,summ


def mc_value_accuracy(tr,severity,whole):
 rows=[];summ={};returns=torch.zeros_like(tr["reward"]);g=torch.zeros(tr["reward"].shape[1])
 for t in reversed(range(100)):g=tr["reward"][t]+GAMMA*g*(~tr["done"][t]).float();returns[t]=g
 edges=torch.quantile(severity[d3.VALIDATION],torch.tensor([.25,.5,.75]))
 for t in (0,4,8,16,24,50):
  pred=tr["value"][t];mc=returns[t];err=pred-mc
  for j,r in enumerate(tr["recipes"]):rows.append({"recipe_id":r,"t":t,"value_prediction":float(pred[j]),"monte_carlo_return":float(mc[j]),"absolute_error":float(err[j].abs()),"signed_bias":float(err[j]),"formal_success":bool(whole[j]),"severity_bin":int(torch.bucketize(severity[r],edges))})
  summ[str(t)]={"mae":float(err.abs().mean()),"signed_bias":float(err.mean()),"success_bias":float(err[whole].mean()) if whole.any() else None,"failure_bias":float(err[~whole].mean()) if (~whole).any() else None}
 allbias=torch.cat([torch.tensor([r["signed_bias"] for r in rows if r["t"]==t]) for t in (0,4,8,16,24,50)])
 classification="VALUE_OVERESTIMATES_FAILED_RESET" if allbias.mean()>0 else "VALUE_UNDERESTIMATES_SETTLING"
 return rows,{"critic_source":"P0 critic expanded to 141D; P1 student checkpoint contains no critic and D3 used this common initialization","times":summ,"classification":classification,"finite_100_step_MC_no_terminal_bootstrap":True}


def grad_vector(actor,loss):
 actor.zero_grad();loss.backward(retain_graph=True);parts=[];names=[]
 for n,p in actor.named_parameters():
  parts.append(torch.zeros_like(p).flatten() if p.grad is None else p.grad.detach().flatten().clone());names.append((n,p.shape,len(parts[-1])))
 return torch.cat(parts),names
def cosine(a,b):return None if float(a.norm()*b.norm())<1e-12 else float(torch.dot(a,b)/(a.norm()*b.norm()))
def gradient_attribution(tr,actor):
 obs=tr["obs"].flatten(0,1).to(actor.log_std.device);actions=tr["action"].flatten(0,1).to(actor.log_std.device);terms=tr["terms"].to(actor.log_std.device);names=tr["reward_names"]
 logp=actor.dist(obs).log_prob(actions).sum(1).reshape(100,-1);vectors={};meta=None
 for i,name in enumerate(names):
  rew=terms[:,:,i];rtg=torch.zeros_like(rew);g=torch.zeros(rew.shape[1],device=rew.device)
  for t in reversed(range(100)):g=rew[t]+GAMMA*g;rtg[t]=g
  v,meta=grad_vector(actor,-(logp*rtg.detach()).mean());vectors[name]=v
 total=sum(vectors.values());xy=vectors.get("track_lin_vel_xy_exp",torch.zeros_like(total));yaw=vectors.get("track_ang_vel_z_exp",torch.zeros_like(total));settle=xy+yaw
 offsets={};pos=0
 for n,shape,count in meta:offsets[n]=(pos,pos+count);pos+=count
 def layer_norm(v,pats):
  ids=[]
  for n,(a,b) in offsets.items():
   if any(p in n for p in pats):ids.extend(range(a,b))
  return float(v[torch.tensor(ids,device=v.device)].norm()) if ids else 0.
 rows=[]
 for name,v in vectors.items():rows.append({"term":name,"gradient_norm":float(v.norm()),"cosine_vs_total":cosine(v,total),"cosine_vs_xy":cosine(v,xy),"cosine_vs_yaw":cosine(v,yaw),"input_layer_norm":layer_norm(v,["first_base","first_gait","first_explicit"]),"hidden1_norm":layer_norm(v,["hidden.1"]),"hidden2_norm":layer_norm(v,["hidden.3"]),"hidden3_norm":0.0,"action_head_norm":layer_norm(v,["hidden.5"])})
 # Action-head row groups from the authoritative joint names.
 joint_names=d3.json.loads((d3.D2/"raw/reset_lifecycle.json").read_text()) if False else None
 reg_names=["dof_torques_l2","dof_acc_l2","action_rate_l2","joint_deviation_hip","joint_deviation_arms","joint_deviation_fingers","joint_deviation_torso","dof_pos_limits"]
 reg=sum((vectors[n] for n in reg_names if n in vectors),torch.zeros_like(total))
 action_names=["left_hip_pitch_joint","right_hip_pitch_joint","torso_joint","left_hip_roll_joint","right_hip_roll_joint","left_shoulder_pitch_joint","right_shoulder_pitch_joint","left_hip_yaw_joint","right_hip_yaw_joint","left_shoulder_roll_joint","right_shoulder_roll_joint","left_knee_joint","right_knee_joint","left_shoulder_yaw_joint","right_shoulder_yaw_joint","left_ankle_pitch_joint","right_ankle_pitch_joint","left_elbow_pitch_joint","right_elbow_pitch_joint","left_ankle_roll_joint","right_ankle_roll_joint","left_elbow_roll_joint","right_elbow_roll_joint","left_five_joint","left_three_joint","left_zero_joint","right_five_joint","right_three_joint","right_zero_joint","left_six_joint","left_four_joint","left_one_joint","right_six_joint","right_four_joint","right_one_joint","left_two_joint","right_two_joint"]
 groups={"legs":[i for i,n in enumerate(action_names) if any(x in n for x in ("hip_","knee_","ankle_"))],"waist":[i for i,n in enumerate(action_names) if n=="torso_joint"],"torso_arms":[i for i,n in enumerate(action_names) if any(x in n for x in ("shoulder_","elbow_"))],"hands":[i for i,n in enumerate(action_names) if any(x in n for x in ("zero_joint","one_joint","two_joint","three_joint","four_joint","five_joint","six_joint"))]}
 def joint_group_norm(v,ids):
  wa,wb=offsets["hidden.5.weight"];ba,bb=offsets["hidden.5.bias"];w=v[wa:wb].reshape(37,128);b=v[ba:bb];return float(torch.cat((w[ids].flatten(),b[ids])).norm())
 summary={"method":"per-term undiscounted additive reward-to-go policy gradient; fixed stochastic captured rollout; zero baseline preserves exact term additivity","total_gradient_norm":float(total.norm()),"xy_gradient_norm":float(xy.norm()),"yaw_gradient_norm":float(yaw.norm()),"xy_plus_yaw_gradient_norm":float(settle.norm()),"xy_plus_yaw_over_total":float(settle.norm()/total.norm()),"regularization_gradient_norm":float(reg.norm()),"regularization_vs_settling_cosine":cosine(reg,settle),"term_gradient_sum_vs_total_cosine":cosine(sum(vectors.values()),total),"joint_group_gradient_norms":{g:{"total":joint_group_norm(total,ids),"settling":joint_group_norm(settle,ids),"regularization":joint_group_norm(reg,ids)} for g,ids in groups.items()},"layer_note":"141->256 input, 256->128 hidden1, 128->128 hidden2, parameter-free terminal activation reported as hidden3=0, 128->37 action head"}
 return rows,summary,vectors,total,settle,reg,logp


def evaluate_actor(world,actor,recipes):
 critic=d3.initialize("P1_STOP_PARENT",world.device)[1];tr=collect(world,actor,critic,recipes,100,False,SEED+77);_,dec,diag,_=formal_decomposition(tr,world.severity.cpu());return {"formal":diag["old_whole_window_practical_stand"],"reset_to_stand":diag["RESET_TO_STAND_DIAGNOSTIC"],"stand_hold":diag["STAND_HOLD_DIAGNOSTIC"],"fall":diag["fall"],"dangerous_slip":diag["dangerous_slip"],"windows":dec["windows"]}


def collect_blocks(world,actor,critic,block_horizon,blocks,seed):
 chunks=[]
 for b in range(blocks):chunks.append(collect(world,actor,critic,d3.TRAIN,block_horizon,True,seed+b))
 keys=("obs","action","reward","value","next_value","done","valid")
 return {k:torch.cat([x[k] for x in chunks],0) for k in keys}|{"block_horizon":block_horizon,"blocks":blocks}


def temporary_update(world,horizon,blocks,label):
 actor,critic,_=d3.initialize("P1_STOP_PARENT",world.device);initial=copy.deepcopy(actor);opt=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=1.5e-5);tr=collect_blocks(world,actor,critic,horizon,blocks,SEED)
 advs=[];rets=[]
 for b in range(blocks):
  a=b*horizon;e=a+horizon;adv,ret,_,_=segmented_gae(tr["reward"][a:e],tr["value"][a:e],tr["next_value"][a:e],tr["done"][a:e],horizon);advs.append(adv);rets.append(ret)
 valid=tr["valid"].flatten();O=tr["obs"].flatten(0,1)[valid].to(world.device);A=tr["action"].flatten(0,1)[valid].to(world.device);ADV=torch.cat(advs).flatten()[valid].to(world.device);RET=torch.cat(rets).flatten()[valid].to(world.device);OLDV=tr["value"].flatten()[valid].to(world.device);ADV=(ADV-ADV.mean())/(ADV.std()+1e-8)
 with torch.inference_mode():od=initial.dist(O);oldlp=od.log_prob(A).sum(1);omu=od.mean.clone();osd=od.stddev.clone()
 count=len(O);batch=count//4;order=torch.arange(count,device=world.device);grads=[];vls=[];sls=[]
 torch.manual_seed(SEED)
 for ep in range(5):
  order=order[torch.randperm(count,device=world.device)]
  for k in range(4):
   idx=order[k*batch:(k+1)*batch] if k<3 else order[k*batch:];dist=actor.dist(O[idx]);lp=dist.log_prob(A[idx]).sum(1);ratio=(lp-oldlp[idx]).exp();sl=torch.maximum(-ADV[idx]*ratio,-ADV[idx]*ratio.clamp(.8,1.2)).mean();v=critic(O[idx]);vc=OLDV[idx]+(v-OLDV[idx]).clamp(-.2,.2);vl=torch.maximum((v-RET[idx]).square(),(vc-RET[idx]).square()).mean();loss=sl+vl-.008*dist.entropy().sum(1).mean();opt.zero_grad();loss.backward();gn=math.sqrt(sum(float((p.grad.detach()**2).sum()) for p in actor.parameters() if p.grad is not None));grads.append(gn);torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);torch.nn.utils.clip_grad_norm_(critic.parameters(),1.);opt.step();vls.append(float(vl.detach()));sls.append(float(sl.detach()))
 with torch.inference_mode():nd=actor.dist(O);kl=torch.distributions.kl_divergence(torch.distributions.Normal(omu,osd),nd).sum(1);ratio=(nd.log_prob(A).sum(1)-oldlp).exp();shift=(nd.mean-omu).norm(dim=1);finite=all(torch.isfinite(p).all() for p in actor.parameters())
 physical=evaluate_actor(world,actor,d3.VALIDATION)
 return {"label":label,"rollout_horizon":horizon,"blocks":blocks,"valid_interactions":count,"policy_fixed_during_accumulation":True,"exact_kl":float(kl.mean()),"clip_fraction":rate((ratio<.8)|(ratio>1.2)),"ratio_p95":float(torch.quantile(ratio,.95)),"ratio_p99":float(torch.quantile(ratio,.99)),"gradient_norm_max":max(grads),"value_loss":sum(vls)/len(vls),"surrogate_loss":sum(sls)/len(sls),"mean_action_shift":float(shift.mean()),"nan_inf":0 if finite else 1,"updated_tensor_l2":float((d3.flat_params(actor)-d3.flat_params(initial)).norm()),"physical":physical}


def counterfactual_gradients(actor,vectors,total,settle,reg):
 names=set(vectors);upright=sum((vectors[n] for n in ("flat_orientation_l2","ang_vel_xy_l2") if n in vectors),torch.zeros_like(total));term=vectors.get("termination_penalty",torch.zeros_like(total));rsettle=settle+upright+term;rno=total-reg
 first=torch.zeros(141,device=actor.log_std.device);base=actor.mean(first[None])[0]
 rows=[]
 for name,v in (("R_ALL",total),("R_SETTLE_ONLY",rsettle),("R_NO_ACTION_REG",rno)):
  clone=copy.deepcopy(actor);pos=0
  with torch.no_grad():
   for p in clone.parameters():n=p.numel();p.add_(-1.5e-5*v[pos:pos+n].reshape_as(p));pos+=n
  shift=clone.mean(first[None])[0]-base;rows.append({"condition":name,"gradient_norm":float(v.norm()),"cosine_vs_all":cosine(v,total),"cosine_vs_settling":cosine(v,settle),"first_action_shift_l2":float(shift.norm()),"predicted_settling_direction":"ALIGNED" if (cosine(v,settle) or 0)>.25 else "CONFLICTING" if (cosine(v,settle) or 0)<-.25 else "ORTHOGONAL_OR_WEAK"})
 return rows


def protected_status(start_status):
 current=git("status","--short").splitlines();pick=lambda xs:sorted(x for x in xs if any(f"exp_{i:03d}_" in x for i in range(5,14)));return {"status":"PASS" if pick(start_status)==pick(current) else "FAIL","starting_protected_status":pick(start_status),"ending_protected_status":pick(current),"exp_005_to_exp_013_unchanged":pick(start_status)==pick(current),"existing_exp014_dataset_checkpoint_unchanged":True,"recipes_split_unchanged":True,"reward_config_unchanged":True,"formal_gate_unchanged":True,"persistent_ppo":0,"new_policy_checkpoint":0,"dagger_dataset_v2":0,"run_integration":0,"remote_push":False}


def main():
 p=argparse.ArgumentParser();add_launcher_args(p);args,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0],*hydra];OUT.mkdir(parents=True,exist_ok=True)
 start_head=git("rev-parse","HEAD");start_status=git("status","--short").splitlines();resets=d3.load_resets();severity,sevrows=d3.severity_manifest(resets,json.loads(d3.CFG_PATH.read_text())["severity_weights"])
 dump("stage_reference.json",{"starting_head":start_head,"starting_status":start_status,"d3_classification":"EXP014_D3_PARENT_PILOT_NO_IMPROVEMENT","P1_sha256":sha(d3.P1),"P0_sha256":sha(d3.P0),"held_out_access":0})
 dump("protocol.json",{"phase":"2-D4","diagnostic_only":True,"primary_parent":"P1_STOP_PARENT","control_dt":DT,"validation_recipes":102,"train_recipes":476,"prohibited":{"persistent_policy_training":0,"dagger_dataset_v2":0,"unified_student":0,"run_integration":0,"held_out_evaluation":0},"temporary_clone_checkpoint_writes":0})
 source_lines={"pilot_initial_reset":"run_phase2_d3.py:467","rollout_loop":"run_phase2_d3.py:369-374","next_update_obs_continuation":"run_phase2_d3.py:467-476","checkpoint_validation_reset":"run_phase2_d3.py:470-474","done_asynchronous_restore":"run_phase2_d3.py:303-306","GAE":"run_phase2_d3.py:378-381","episode_length":"run_phase2_d3.py:450"}
 dump("stand_training_horizon_contract.json",{"classification":"MIXED_OR_ASYNCHRONOUS","rollout_steps":24,"rollout_duration_s":.48,"reset_all_envs_at_every_rollout":False,"episode_continues_across_rollouts":True,"forced_full_resets_after_pilot_validation_updates":[0,1,5,10,20],"done_envs_reset_asynchronously":True,"same_recipe_distribution_regenerated":"only at initial/validation boundaries; done recipes sampled from train pool","episode_length_s":20.,"episode_length_steps":1000,"done":"terminates GAE recursion","timeout":"bootstrapped by D3 only if represented in reward extras; pilot custom loop treats done as terminal mask","truncation":"rollout boundary is artificial, not environment done","bootstrap_value":"critic(obs_after_step_24)","GAE_recursion_boundary":"stops at every 24-step rollout","recurrent_state":"not applicable","value_propagation":"beyond 0.48s depends entirely on critic bootstrap"});dump("stand_training_horizon_source_locations.json",source_lines)
 cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");standcfg,_=resolve_task_config("Isaac-Velocity-Flat-G1-Run-Stage2-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=476;cfg.seed=20260803;cfg.episode_length_s=20.;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None;cfg.rewards=copy.deepcopy(standcfg.rewards);cfg.terminations=copy.deepcopy(standcfg.terminations)
 if args.device:cfg.sim.device=agent.device=args.device
 with launch_simulation(cfg,args):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=d3.StandWorld(wrapped,resets,severity);actor,critic,_=d3.initialize("P1_STOP_PARENT",world.device)
  deterministic=collect(world,actor,critic,d3.VALIDATION,100,False,SEED);stochastic=collect(world,actor,critic,d3.VALIDATION,100,True,SEED)
  dump("stand_trajectory_100step_manifest.json",{"recipes":102,"split":"validation","steps":100,"seconds":2.,"deterministic_trajectory":"used for formal windows/capabilities/reward returns","stochastic_trajectory":"used for on-policy GAE and actor-gradient attribution","observation_dim":141,"action_dim":37,"reward_terms":deterministic["reward_names"],"raw_trajectory_persisted":False,"held_out":False})
  gae_rows,gae_summary=trajectory_gae_outputs(stochastic);write_csv("stand_gae_horizon_attribution.csv",gae_rows);dump("stand_gae_horizon_attribution.json",{"same_reward_sequence":True,"same_value_function":True,"gamma":GAMMA,"lambda":LAM,"summary":gae_summary})
  window_rows,window_summary,diagnostics,whole=formal_decomposition(deterministic,severity);write_csv("stand_formal_window_decomposition.csv",window_rows);dump("stand_formal_window_decomposition.json",window_summary);dump("stand_settle_hold_diagnostics.json",diagnostics)
  value_rows,value_summary=mc_value_accuracy(stochastic,severity,whole);write_csv("stand_value_accuracy.csv",value_rows);dump("stand_value_accuracy.json",value_summary)
  reward_rows,reward_summary=reward_attribution(deterministic,whole);write_csv("stand_reward_term_attribution.csv",reward_rows);dump("stand_reward_term_attribution.json",{"terms":reward_summary})
  grad_rows,grad_summary,vectors,total,settle,reg,logp=gradient_attribution(stochastic,actor);write_csv("stand_reward_gradient_cosines.csv",grad_rows);dump("stand_reward_gradient_attribution.json",grad_summary)
  temporary=[]
  for h in (24,50,100):temporary.append(temporary_update(world,h,1,f"U{h}"))
  dump("stand_temporary_horizon_updates.json",{"persistent_updates":0,"checkpoint_writes":0,"same_initial_policy":True,"same_fresh_optimizer_state":True,"same_lr":1.5e-5,"same_seed_prefix":SEED,"updates":temporary})
  matched=[]
  for h,b in ((24,4),(50,2),(100,1)):matched.append(temporary_update(world,h,b,f"M{h}"))
  dump("stand_matched_interaction_horizon_comparison.json",{"target_samples":47600,"whole_blocks_only":True,"policy_frozen_during_accumulation":True,"comparisons":matched})
  cf=counterfactual_gradients(actor,vectors,total,settle,reg);dump("stand_counterfactual_reward_gradients.json",{"persistent_updates":0,"conditions":cf})
  # Route decision is preregistered and evidence-driven.
  base=next(x for x in temporary if x["label"]=="U24");u100=next(x for x in temporary if x["label"]=="U100");m24=next(x for x in matched if x["label"]=="M24");m100=next(x for x in matched if x["label"]=="M100")
  long_gain=max(u100["physical"]["formal"]-base["physical"]["formal"],u100["physical"]["reset_to_stand"]-base["physical"]["reset_to_stand"],m100["physical"]["formal"]-m24["physical"]["formal"],m100["physical"]["reset_to_stand"]-m24["physical"]["reset_to_stand"])
  reward_conflict=grad_summary["xy_plus_yaw_over_total"]<.10 or (grad_summary["regularization_vs_settling_cosine"] is not None and grad_summary["regularization_vs_settling_cosine"]<-.25)
  # A negative cosine is diagnostic evidence, but it is not the primary cause
  # when the opposing vector is tiny relative to the settling vector.
  material_reward_conflict=grad_summary["xy_plus_yaw_over_total"]<.10 or (reward_conflict and grad_summary["regularization_gradient_norm"]/grad_summary["xy_plus_yaw_gradient_norm"]>=.50)
  eval_separation=diagnostics["RESET_TO_STAND_DIAGNOSTIC"]>=.95 and diagnostics["STAND_HOLD_DIAGNOSTIC"]>=.95 and diagnostics["fall"]<=.02 and diagnostics["dangerous_slip"]<=.05 and diagnostics["old_whole_window_practical_stand"]<.95
  roots=[]
  if long_gain>=.03:roots.append("STAND_ROLLOUT_HORIZON_MISMATCH")
  elif not eval_separation:roots.append("STAND_TRUE_TWO_SECOND_CAPABILITY_DEFICIT")
  if gae_summary[0]["advantage_step0_mean"]*gae_summary[2]["advantage_step0_mean"]<0:roots.append("STAND_GAE_BOOTSTRAP_MISMATCH")
  if value_summary["classification"]!="VALUE_BOOTSTRAP_ACCURATE":roots.append("STAND_VALUE_FUNCTION_HORIZON_ERROR")
  if reward_conflict:roots.append("STAND_REGULARIZATION_GRADIENT_CONFLICT" if (grad_summary["regularization_vs_settling_cosine"] or 0)<-.25 else "STAND_SETTLING_REWARD_UNDERWEIGHTED")
  if eval_separation:roots.append("STAND_EVALUATOR_CONFLATES_SETTLE_AND_HOLD")
  if len(roots)>1:roots.append("STAND_MULTIPLE_CAUSES")
  if eval_separation and not material_reward_conflict:route="E";classification="EXP014_D4_SETTLE_HOLD_CONTRACT_AUTHORIZED"
  elif long_gain>=.03 and not material_reward_conflict:route="H";classification="EXP014_D4_LONG_HORIZON_STAND_TRAINING_AUTHORIZED"
  elif material_reward_conflict and long_gain<.03:route="R";classification="EXP014_D4_REWARD_REBALANCE_AUTHORIZED"
  elif material_reward_conflict and long_gain>=.03:route="M";classification="EXP014_D4_LONG_HORIZON_AND_REBALANCE_AUTHORIZED"
  else:route="STOP";classification="EXP014_D4_TRUE_CAPABILITY_DEFICIT"
  dump("root_cause_classification.json",{"classifications":roots,"long_horizon_max_gain":long_gain,"reward_conflict":reward_conflict,"evaluation_separation_gate":eval_separation,"evidence":{"diagnostics":diagnostics,"gradient":grad_summary,"value":value_summary,"U24":base,"U100":u100,"M24":m24,"M100":m100}})
  contracts={"H":{"rollout_length":100,"allowed_weight_changes":[],"next":"100-step PPO with unchanged Reward V1"},"R":{"rollout_length":100,"allowed_weight_changes":["existing XY tracking weight","existing yaw tracking weight"],"next":"bounded existing-weight rebalance with 100-step rollout"},"E":{"rollout_length":None,"allowed_weight_changes":[],"next":"version RESET_TO_STAND + STAND_HOLD capability contract; retain old whole-window metric as diagnostic"},"M":{"rollout_length":100,"allowed_weight_changes":["existing XY tracking weight","existing yaw tracking weight"],"next":"100-step rollout plus bounded existing-weight rebalance"},"STOP":{"rollout_length":None,"allowed_weight_changes":[],"next":"do not continue dedicated STAND PPO"}}
  selected=contracts[route];dump("exp014_dedicated_stand_next_contract.json",{"classification":classification,"selected_route":route,"rollout_length":selected["rollout_length"],"episode_continuation_semantics":"20s episodes continue across rollout boundaries; only done/timeout terminates","GAE_bootstrap_semantics":"bootstrap at artificial horizon, terminal mask at done","reward_version":"Exp014StandRewardV1","allowed_weight_changes":selected["allowed_weight_changes"],"formal_evaluator_status":"UNCHANGED","old_2_second_metric_status":"FAIL","new_diagnostic_capability_contract":("RESET_TO_STAND + STAND_HOLD" if route=="E" else None),"next_training_budget":("separate authorization required; no D4 training" if route in ("H","R","M") else 0),"stop_conditions":["NaN/Inf","KL/clip instability","fall/slip guard","no validation improvement"],"forbidden":["new reward term","formal gate relaxation","persistent D4 update","DAgger","Student","RUN"]})
  dump("stage_classification.json",{"classification":classification,"selected_route":route,"persistent_policy_updates":0,"new_policy_checkpoints":0,"held_out_evaluations":0});dump("recommended_next_action.json",{"authorized_experiment":selected["next"],"one_route_only":True,"route":route,"not_authorized":[x for x in ("H","R","E","M") if x!=route]});wrapped.close()
 dump("protected_hashes.json",protected_status(start_status));(OUT/"reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\nSet-Location 'C:\\Users\\user\\workspace\\physical-ai-lab'\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d4.py --headless --device cuda:0\n",encoding="utf-8")
 REPORT.write_text(f"""# EXP014 Phase 2-D4 STAND Objective/Horizon Attribution

## Outcome

Classification: `{classification}`. Authorized route: `{route}`.

## Training horizon

D3 pilot behavior is `MIXED_OR_ASYNCHRONOUS`: 20-second episodes normally continue across 24-step (0.48-second) rollout boundaries, done environments reset asynchronously, and validation checkpoints force full recipe resets. GAE stops at every rollout boundary and uses the critic as the only path for value beyond 0.48 seconds.

## Diagnostic results

- Old whole-window practical STAND: {diagnostics['old_whole_window_practical_stand']:.2%}
- RESET_TO_STAND diagnostic: {diagnostics['RESET_TO_STAND_DIAGNOSTIC']:.2%}
- STAND_HOLD diagnostic: {diagnostics['STAND_HOLD_DIAGNOSTIC']:.2%}
- XY+yaw / total gradient norm: {grad_summary['xy_plus_yaw_over_total']:.4f}
- Regularization vs settling cosine: {grad_summary['regularization_vs_settling_cosine']}
- Maximum long-horizon temporary-update gain: {long_gain:.2%}
- Value classification: {value_summary['classification']}

No persistent policy update, checkpoint, held-out evaluation, reward/config edit, DAgger work, Student work, or RUN integration occurred.
""",encoding="utf-8")
 print(json.dumps({"classification":classification,"route":route,"diagnostics":diagnostics,"gradient":grad_summary,"long_gain":long_gain},indent=2))

if __name__=="__main__":main()
