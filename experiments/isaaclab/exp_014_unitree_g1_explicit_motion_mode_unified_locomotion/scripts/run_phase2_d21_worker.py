"""Isaac worker for D21 identity-complete capture and temporary causal probes."""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, io, json, math, os, random, sys
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch

HERE=Path(__file__).resolve();REPO=HERE.parents[4]
OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d21_identity_complete_support_causality";RAW=OUT/"raw"
SEED=20279501;DT=.02;N=64;STEPS=100

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d18=module("d18_d21",HERE.parent/"run_phase2_d18_precursor.py");d16=d18.d16;d15=d18.d15;d6=d18.d6
reconstruct_mod=module("d21_reconstruct_worker",HERE.parent/"d21_reward_reconstruction.py")
from g1_explicit_motion_mode.contract import MotionMode,minimum_jerk
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def thash(m):return d16.thash(m)
def ohash(x):
 b=io.BytesIO();torch.save(x,b);return hashlib.sha256(b.getvalue()).hexdigest()
def ahash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def t2n(x):return x.detach().cpu().numpy()
def schedule(t,peak=.7):
 target=peak*minimum_jerk((t/.35).clamp(0,1));target=torch.where(t<.75,target,torch.zeros_like(t))
 env=torch.where(t<.50,torch.ones_like(t),torch.where(t<.75,1-minimum_jerk((t-.50)/.25),torch.zeros_like(t)))
 velw=torch.where(t<=.20,torch.full_like(t,.15),torch.where(t<.50,.15+.45*minimum_jerk((t-.20)/.30),torch.where(t<.75,.60+.40*minimum_jerk((t-.50)/.25),torch.ones_like(t))))
 yaww=torch.where(t<=.20,torch.full_like(t,.25),torch.where(t<.50,.25+.50*minimum_jerk((t-.20)/.30),torch.ones_like(t)))
 unload=torch.where((t>=.20)&(t<=.60),torch.ones_like(t),torch.zeros_like(t));return target,env,velw,yaww,unload
def euler_xy(q):
 w,x,y,z=q.unbind(-1);roll=torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y));pitch=torch.asin((2*(w*y-z*x)).clamp(-1,1));return torch.stack((roll,pitch),-1)
def snap_hash(pool,i):
 h=hashlib.sha256()
 for k,v in sorted(pool["snapshot"].items()):h.update(k.encode());h.update(v[i].contiguous().numpy().tobytes())
 return h.hexdigest()
def returns(r,done):
 out=torch.zeros_like(r);run=torch.zeros(r.shape[1],device=r.device)
 for t in reversed(range(len(r))):run=r[t]+.99*run*(~done[t]).float();out[t]=run
 return out

def capture(world,policy,critic,train,scales,weights):
 picks=list(range(N));d18.restore_n(world,train,picks);world.configure(scales,.7,weights);world.target[:N]=0;world.target[:N,0]=.3;world.age[:]=0;world.advance_command();obs=world.obs()
 arrays={};lists={}
 def put(k,v):lists.setdefault(k,[]).append(t2n(v))
 prev_lz=None;prev_yaw=world.robot.data.root_ang_vel_w[:N,2].detach().clone();ss=[torch.zeros(N,dtype=torch.long,device=world.device) for _ in range(3)]
 recipes=torch.tensor(train["recipes"][:N],device=world.device,dtype=torch.long);snapshot=torch.arange(N,device=world.device);rollout=snapshot.clone()
 snapshot_hashes=np.asarray([snap_hash(train,i) for i in picks],dtype="U64")
 source_obs_hashes=[]
 for step in range(STEPS):
  if step==0:
   source_obs_hashes=[hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in t2n(obs[:N])]
  cpu_state=torch.random.get_rng_state();device_obj=torch.device(world.device);cuda_state=torch.cuda.get_rng_state(device_obj) if device_obj.type=="cuda" else torch.empty(0,dtype=torch.uint8)
  with torch.inference_mode():
   dist=policy.residual.dist(obs);raw=dist.sample();base=policy.base_action(obs);resmean=policy.residual.net(obs);meanfinal=policy.mean_action(obs);sampled=policy.action(obs,raw);lp=dist.log_prob(raw).sum(1);value=critic(obs)
  prev_action=world.env.action_manager.prev_action.clone();curcmd=world.state.physical_command.clone();prevcmd=world.state.previous_physical_command.clone();gate=policy.gate(obs)
  _,_,done,extras=world.wrapped.step(sampled);timeout=extras.get("time_outs",torch.zeros_like(done)).bool();fall=done.bool()&~timeout
  p=d18.privileged(world,N,prev_lz);prev_lz=p["Lz"].detach();yaw_now=world.robot.data.root_ang_vel_w[:N,2];yaw_acc=(yaw_now-prev_yaw)/DT;prev_yaw=yaw_now.detach().clone()
  force=world.sensor.data.net_forces_w_history[:N,-1,world.sf,:];fnorm=force.norm(dim=-1);support_valid=(fnorm>5).any(1);feet=world.robot.data.body_lin_vel_w[:N,world.rf,:2];bad=((feet.norm(dim=-1)>.55)&(fnorm>5)).any(1);world.slip_streak[:N]=torch.where(bad,world.slip_streak[:N]+1,torch.zeros_like(world.slip_streak[:N]));slip=world.slip_streak[:N]>=5;impact=fnorm.amax(1)>3500
  vr=world.robot.data.joint_vel[:N].abs().div(world.limits[:N].clamp_min(1e-6)).amax(1);eff=world.robot.data.joint_effort_limits[:N].abs().clamp_min(1e-6);tr=world.robot.data.applied_torque[:N].abs().div(eff).amax(1);world.vsat_streak[:N]=torch.where(vr>.95,world.vsat_streak[:N]+1,torch.zeros_like(world.vsat_streak[:N]));world.tsat_streak[:N]=torch.where(tr>.95,world.tsat_streak[:N]+1,torch.zeros_like(world.tsat_streak[:N]));vs=world.vsat_streak[:N]>=5;ts=world.tsat_streak[:N]>=5
  time=torch.full((N,),step*DT,device=world.device);target,env,velw,yaww,unload=schedule(time);load=torch.exp(-((p["load_imbalance"]-target)/scales["sigma_load"]).square())*support_valid;total=torch.exp(-((p["support_ratio"]-1)/scales["sigma_support"]).square());unloadr=torch.exp(-((p["low_load_ratio"]-scales["unload_target"])/scales["sigma_unload"]).square())
  loadterm=env*load;totalterm=env*total;slipterm=env*(-.2*p["support_slip"]);unloadterm=unload*unloadr
  preventive_env=torch.where(time<=.50,torch.ones_like(time),torch.where(time<.75,1-minimum_jerk((time-.50)/.25),torch.zeros_like(time)));preventive=preventive_env*(torch.exp(-(p["Lz"]/scales["sigma_Lz"]).square())+torch.exp(-(p["dLz_dt"]/scales["sigma_dLz"]).square())+torch.exp(-(p["contact_yaw_moment"]/scales["sigma_Mz"]).square()))
  vel=world.robot.data.root_lin_vel_b[:N,:2];yaw=world.robot.data.root_ang_vel_b[:N,2];ve=(vel-torch.tensor([.3,0.],device=world.device)).norm(dim=1);ye=yaw.abs();velocity=6*velw*torch.exp(-ve.square()/.25);yawtrack=8*yaww*torch.exp(-ye.square()/.25)
  gravity=world.robot.data.projected_gravity_b[:N];upright=(-gravity[:,2]).clamp(-1,1);vertical=world.robot.data.root_lin_vel_b[:N,2];acc=((world.robot.data.joint_vel[:N]-world.previous_joint_vel[:N])/DT).square().sum(1);world.previous_joint_vel[:N]=world.robot.data.joint_vel[:N].clone();torque=world.robot.data.applied_torque[:N].square().sum(1);actionrate=(sampled[:N]-prev_action[:N]).square().sum(1);resmag=(policy.bound*torch.tanh(raw[:N])).square().sum(1)
  uprightterm=2*torch.exp(-(1-upright).square()/.1);termination=-200*fall[:N].float();safetyrest=-.2*vertical.square()-slip.float()-impact.float()-vs.float()-ts.float();regular=-2e-6*torque-1e-7*acc-.005*actionrate-.02*resmag
  online=weights["preventive"]*preventive+weights["support"]*(loadterm+totalterm+slipterm+unloadterm)+weights["tracking"]*(velocity+yawtrack)+weights["safety"]*(uprightterm+termination+safetyrest+regular)
  world.age+=1;world.advance_command();nextobs=world.obs()
  with torch.inference_mode():nextvalue=critic(nextobs)
  obs124=torch.cat((obs[:N,:123],torch.zeros(N,1,device=world.device)),1)
  fixed={"rollout_id":rollout,"snapshot_id":snapshot,"recipe_id":recipes,"environment_index":torch.arange(N,device=world.device),"control_step":torch.full((N,),step,device=world.device,dtype=torch.long),"time_since_start":time,"rng_state_index":torch.full((N,),step,device=world.device,dtype=torch.long)}
  for k,v in fixed.items():put(k,v)
  for k,v in {"obs_141":obs[:N],"obs_124":obs124,"base_mean_action":base[:N],"residual_output":resmean[:N],"final_mean_action":meanfinal[:N],"sampled_action":sampled[:N],"sampled_raw_residual":raw[:N],"log_probability":lp[:N],"actor_std":dist.stddev[:N],"actor_log_std":policy.residual.log_std[None].expand(N,-1),"critic_observation":obs[:N],"value_prediction":value[:N],"next_value_prediction":nextvalue[:N],"done":done[:N],"timeout":timeout[:N],"previous_action":prev_action[:N],"current_command":curcmd[:N],"previous_command":prevcmd[:N],"command_delta":curcmd[:N]-prevcmd[:N],"motion_mode":world.state.target_mode[:N],"previous_mode":world.state.previous_target_mode[:N],"ramp_progress":obs[:N,140],"start_gate":gate[:N],"root_pose":torch.cat((world.robot.data.root_pos_w[:N],world.robot.data.root_quat_w[:N]),1),"root_velocity":torch.cat((world.robot.data.root_lin_vel_w[:N],world.robot.data.root_ang_vel_w[:N]),1),"joint_position":world.robot.data.joint_pos[:N],"joint_velocity":world.robot.data.joint_vel[:N],"contact_force":force,"foot_tangential_velocity":feet,"F_L":p["F_L"],"F_R":p["F_R"],"F_total":p["F_total"],"signed_load_balance":(p["F_L"]-p["F_R"])/(p["F_total"]+1e-6),"unsigned_load_balance":p["load_imbalance"],"low_load_ratio":p["low_load_ratio"],"support_valid":support_valid,"Lz":p["Lz"],"dLz_dt":p["dLz_dt"],"contact_yaw_moment":p["contact_yaw_moment"],"yaw_rate":yaw,"yaw_acceleration":yaw_acc,"roll_pitch":euler_xy(world.robot.data.root_quat_w[:N]),"projected_gravity":gravity,"upright_scalar":upright,"pelvis_vertical_velocity":vertical,"fall":fall[:N],"dangerous_slip":slip,"impact":impact,"velocity_saturation":vs,"torque_saturation":ts,"target_load":target,"support_envelope":env,"velocity_envelope":velw,"yaw_envelope":yaww,"unload_envelope":unload,"load_reward_gpu":loadterm,"total_support_reward_gpu":totalterm,"support_slip_reward_gpu":slipterm,"swing_unload_reward_gpu":unloadterm,"preventive_yaw_reward_gpu":preventive,"velocity_reward_gpu":velocity,"yaw_reward_gpu":yawtrack,"upright_reward_gpu":uprightterm,"termination_reward_gpu":termination,"safety_rest_reward_gpu":safetyrest,"regularization_reward_gpu":regular,"online_reward_gpu":online,"velocity_error":ve,"yaw_error":ye,"support_ratio":p["support_ratio"],"support_foot_slip":p["support_slip"],"torque_sq":torque,"joint_acc_sq":acc,"action_rate_sq":actionrate,"residual_mag_sq":resmag}.items():put(k,v)
  lists.setdefault("cpu_rng_state",[]).append(t2n(cpu_state));lists.setdefault("cuda_rng_state",[]).append(t2n(cuda_state))
  obs=nextobs
 arrays={k:np.stack(v) for k,v in lists.items()};arrays["source_snapshot_hashes"]=snapshot_hashes;arrays["source_observation_hashes"]=np.asarray(source_obs_hashes,dtype="U64");canonical=reconstruct_mod.reconstruct(arrays,scales,weights);arrays.update(canonical)
 return arrays

def term_rewards(a,device):
 T=lambda k:torch.from_numpy(a[k]).to(device)
 w=json.loads((REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d18_early_support_yaw_objective/reward_gradient_calibration.json").read_text())["deterministic_proportional_scales"]
 upterm=w["safety"]*(T("upright_reward")+T("termination_reward"));sup=w["support"]
 signed=T("signed_load_balance");env=T("support_envelope");valid=T("support_valid");sigma=.30000001192092896
 left=env*valid*torch.exp(-((signed-.7)/sigma).square())*sup+upterm+sup*T("total_support_reward")
 right=env*valid*torch.exp(-((signed+.7)/sigma).square())*sup+upterm+sup*T("total_support_reward")
 return {
 "Q_LOAD_ABS_R1":sup*T("load_reward")+upterm,"Q_TOTAL_SUPPORT":sup*T("total_support_reward")+upterm,"Q_SUPPORT_SLIP":sup*T("support_slip_reward")+upterm,"Q_SWING_UNLOAD":sup*T("swing_unload_reward")+upterm,
 "Q_LOAD_UNLOAD_R1":sup*(T("load_reward")+T("swing_unload_reward"))+upterm,"Q_LOAD_SUPPORT_R1":sup*(T("load_reward")+T("total_support_reward"))+upterm,"Q_SUPPORT_FULL_R1":sup*(T("load_reward")+T("total_support_reward")+T("support_slip_reward")+T("swing_unload_reward"))+upterm,
 "Q_PREVENTIVE_YAW_CONTROL":w["preventive"]*T("preventive_yaw_reward")+upterm,"Q_SIGN_LEFT_R1":left,"Q_SIGN_RIGHT_R1":right}

def gradients(policy,a,rewards):
 device=next(policy.parameters()).device;obs=torch.from_numpy(a["obs_141"]).to(device);raw=torch.from_numpy(a["sampled_raw_residual"]).to(device);done=torch.from_numpy(a["done"]).to(device).bool();active=torch.from_numpy(a["start_gate"]).to(device)>0;out={};vectors={}
 for name,r in rewards.items():
  ret=returns(r,done);adv=ret-ret.mean(1,keepdim=True);av=adv[active];adv=(adv-av.mean())/(av.std()+1e-8);dist=policy.residual.dist(obs.flatten(0,1));lp=dist.log_prob(raw.flatten(0,1)).sum(1);loss=-(lp[active.flatten()]*adv.flatten()[active.flatten()].detach()).mean();params=list(policy.residual.named_parameters());g=torch.autograd.grad(loss,[p for _,p in params],retain_graph=True,allow_unused=True);vectors[name]=torch.cat([(x if x is not None else torch.zeros_like(p)).flatten() for x,(_,p) in zip(g,params)]).detach();layer={n:float((x if x is not None else torch.zeros_like(p)).norm()) for x,(n,p) in zip(g,params)};last=next((x for x,(n,p) in zip(g,params) if n.endswith("net.6.weight")),None);joint=[] if last is None else [float(last[j].norm()) for j in range(last.shape[0])];out[name]={"actor_gradient_norm":float(vectors[name].norm()),"layerwise_norm":layer,"joint_output_norm":joint}
 names=list(vectors);cos={x:{y:float(torch.nn.functional.cosine_similarity(vectors[x][None],vectors[y][None])) for y in names} for x in names}
 return out,cos,vectors

def update_clone(base_policy,base_critic,a,reward):
 p=copy.deepcopy(base_policy);c=copy.deepcopy(base_critic);device=next(p.parameters()).device;obs=torch.from_numpy(a["obs_141"]).to(device);raw=torch.from_numpy(a["sampled_raw_residual"]).to(device);done=torch.from_numpy(a["done"]).to(device).bool();oldlp=torch.from_numpy(a["log_probability"]).to(device);active=torch.from_numpy(a["start_gate"]).to(device)>0;ret=returns(reward,done);dist=p.residual.dist(obs.flatten(0,1));lp=dist.log_prob(raw.flatten(0,1)).sum(1).reshape_as(oldlp);adv=ret-c(obs.flatten(0,1)).reshape_as(ret).detach();av=adv[active];adv[active]=(av-av.mean())/(av.std()+1e-8);actor=-(lp[active]*adv[active].detach()).mean();value=c(obs.flatten(0,1)).reshape_as(ret);critic=.5*(value-ret.detach()).square().mean();entropy=dist.entropy().sum(1).mean();loss=actor+critic-.008*entropy;opt=torch.optim.Adam(list(p.residual.parameters())+list(c.parameters()),lr=1.5e-5);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(p.residual.parameters())+list(c.parameters()),10);opt.step();finite=all(torch.isfinite(x).all() for x in list(p.parameters())+list(c.parameters()));
 with torch.inference_mode():
  oldd=base_policy.residual.dist(obs.flatten(0,1));newd=p.residual.dist(obs.flatten(0,1));kl=torch.distributions.kl_divergence(oldd,newd).sum(1);ratio=(newd.log_prob(raw.flatten(0,1)).sum(1)-oldlp.flatten()).exp();shift=(p.mean_action(obs.flatten(0,1))-base_policy.mean_action(obs.flatten(0,1))).norm(dim=1)
 return p,c,{"actor_loss":float(actor),"critic_loss":float(critic),"finite":finite,"actor_tensor_hash":thash(p.residual),"critic_tensor_hash":thash(c),"exact_kl":float(kl.mean()),"all_step_kl":float(kl.max()),"clip_fraction":float(((ratio<.8)|(ratio>1.2)).float().mean()),"mean_final_action_shift":float(shift.mean()),"residual_bound_compliance":1.0}

def eval_policy(world,policy,train,scales,name):
 d18.restore_n(world,train,list(range(N)));world.configure(scales,.7,{});world.target[:N]=0;world.target[:N,0]=.3;world.age[:]=0;prev=None;ys=[];lz=[];dlz=[];mz=[];le=[];signed=[];total=[];slipm=[];low=[];sides=[];flags=[torch.zeros(N,dtype=torch.bool,device=world.device) for _ in range(5)];ss=[torch.zeros(N,dtype=torch.long,device=world.device) for _ in range(3)]
 for step in range(32):
  world.advance_command();obs=world.obs();
  with torch.inference_mode():action=policy.mean_action(obs)
  _,_,done,extras=world.wrapped.step(action);p=d18.privileged(world,N,prev);prev=p["Lz"].detach();sf=d15.safety(world,N,done,extras,ss)
  for x,y in zip(flags,sf[:5]):x|=y
  target=schedule(torch.full((N,),step*DT,device=world.device))[0];le.append((p["load_imbalance"]-target).abs());signed.append((p["F_L"]-p["F_R"])/(p["F_total"]+1e-6));total.append((p["support_ratio"]-1).abs());slipm.append(p["support_slip"]);low.append(p["low_load_ratio"]);y=world.robot.data.root_ang_vel_b[:N,2];ys.append(y);lz.append(p["Lz"].abs());dlz.append(p["dLz_dt"].abs());mz.append(p["contact_yaw_moment"].abs());valid=(world.sensor.data.net_forces_w_history[:N,-1,world.sf,:].norm(dim=-1)>5);imb=p["load_imbalance"];side=torch.where(~valid.any(1),torch.full_like(imb,3,dtype=torch.long),torch.where(imb<.2,torch.full_like(imb,2,dtype=torch.long),torch.where(p["F_L"]>=p["F_R"],torch.zeros_like(imb,dtype=torch.long),torch.ones_like(imb,dtype=torch.long))));sides.append(side);world.age+=1
 y=torch.stack(ys);sg=torch.stack(signed);sd=torch.stack(sides);dom=sd<2;first=torch.where(dom,torch.arange(32,device=world.device)[:,None],torch.full((32,N),99,device=world.device)).amin(0);rev=(((sd[1:]!=sd[:-1])&(sd[1:]<2)&(sd[:-1]<2))).sum(0);signchanges=((y[1:]*y[:-1])<0).sum(0)
 target_sign=.7 if name=="Q_SIGN_LEFT_R1" else -.7 if name=="Q_SIGN_RIGHT_R1" else None
 return {"episodes":N,"load_target_error":float(torch.stack(le).mean()),"signed_load_balance_mean":float(sg.mean()),"unsigned_load_balance_mean":float(sg.abs().mean()),"signed_target_error":None if target_sign is None else float((sg-target_sign).abs().mean()),"signed_left_target_error":float((sg-.7).abs().mean()),"signed_right_target_error":float((sg+.7).abs().mean()),"total_support_error":float(torch.stack(total).mean()),"support_foot_slip":float(torch.stack(slipm).mean()),"swing_low_load_ratio":float(torch.stack(low).mean()),"yaw_p95":float(torch.quantile(y.abs(),.95)),"yaw_sign_changes":float(signchanges.float().mean()),"Lz_p95":float(torch.quantile(torch.stack(lz),.95)),"dLz_dt_p95":float(torch.quantile(torch.stack(dlz),.95)),"contact_yaw_moment_p95":float(torch.quantile(torch.stack(mz),.95)),"first_dominant_support_step":float(first[first<99].float().mean()) if (first<99).any() else None,"dominant_support_duration":float(dom.float().sum(0).mean()),"support_side_reversal_count":float(rev.float().mean()),"fall":float(flags[0].float().mean()),"dangerous_slip":float(flags[1].float().mean()),"impact":float(flags[2].float().mean()),"velocity_saturation":float(flags[3].float().mean()),"torque_saturation":float(flags[4].float().mean()),"support_yaw_sign_correlation":float(torch.corrcoef(torch.stack((rev.float(),signchanges.float())))[0,1]) if rev.float().std()>0 and signchanges.float().std()>0 else 0.0}

def main():
 parser=argparse.ArgumentParser();add_launcher_args(parser);args,hydra=setup_preset_cli(parser);sys.argv=[sys.argv[0],*hydra];cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=N;cfg.seed=SEED;cfg.episode_length_s=3.;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None
 if args.device:cfg.sim.device=agent.device=args.device
 torch.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED);train=torch.load(d16.RAW/"train_start_snapshots.pt",map_location="cpu",weights_only=False);manifest=json.loads((d16.OUT/"train_start_snapshot_manifest.json").read_text());assert len(train["recipes"])>=N and all(train["valid"][:N]) and manifest["snapshot_tensor_hash"]=="c249014cf79ebeae293d095cfbf8c5706f4f2c5e0e70113655bb791a96b63214"
 scales=json.loads((d18.OUT/"reference_distribution_manifest.json").read_text())["derived_scales"];weights=json.loads((d18.OUT/"reward_gradient_calibration.json").read_text())["deterministic_proportional_scales"]
 ckpt=d16.CKPT/"model_000.pt";saved=torch.load(ckpt,map_location="cpu",weights_only=False)
 with launch_simulation(cfg,args):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=d18.D18World(wrapped,d18.d3.load_resets(),train);policy=d16.StartPolicy(.5).to(world.device);critic=d16.Critic().to(world.device);policy.base.load_state_dict(saved["base_state_dict"]);policy.residual.load_state_dict(saved["residual_state_dict"]);critic.load_state_dict(saved["critic_state_dict"])
  identity={"base_checkpoint_sha":d16.WMOVE_SHA,"d16_initial_checkpoint_sha":sha(ckpt),"actor_hash":thash(policy.residual),"critic_hash":thash(critic),"std_hash":ahash(t2n(policy.residual.log_std)),"normalizer_hash":"NOT_PRESENT","optimizer_state_hash":ohash(saved["optimizer_state_dict"]),"initial_residual_strict_zero":float(policy.residual.net[-1].weight.abs().max())==0 and float(policy.residual.net[-1].bias.abs().max())==0,"initial_parity":d16.parity(policy,world.device)}
  if os.environ.get("D21_PROBE_ONLY")=="1":
   bundle=OUT/"reference_rollout_bundle.npz"
   with np.load(bundle,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
   bundle_sha=sha(bundle)
  else:
   arrays=capture(world,policy,critic,train,scales,weights);RAW.mkdir(parents=True,exist_ok=True);tmp=RAW/"reference_rollout_bundle.tmp";
   with tmp.open("wb") as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
   bundle_sha=sha(tmp);meta={"tmp_path":str(tmp),"sha256":bundle_sha,"transitions":N*STEPS,"identity":identity,"array_count":len(arrays),"sample_order":"control_step_major_then_environment_index"};print("D21_CAPTURE_READY "+json.dumps(meta),flush=True)
   if sys.stdin.readline().strip()!="CONTINUE":raise RuntimeError("parent did not acknowledge durable capture")
  rewards=term_rewards(arrays,world.device);grad,cos,vectors=gradients(policy,arrays,rewards);baseline=eval_policy(world,policy,train,scales,"BASELINE");probes={};temps={}
  for name,r in rewards.items():
   clone,cclone,tm=update_clone(policy,critic,arrays,r);probes[name]=eval_policy(world,clone,train,scales,name);temps[name]=tm
  mirror=json.loads((REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d19_support_objective_symmetry_audit/mirror_contract.json").read_text());gl=vectors["Q_SIGN_LEFT_R1"];gr=vectors["Q_SIGN_RIGHT_R1"];ga=vectors["Q_LOAD_ABS_R1"];symaxis=(gl-gr)/(gl-gr).norm().clamp_min(1e-12);shared=(gl+gr)/(gl+gr).norm().clamp_min(1e-12);sym={"note":"D19 action mirror is applied to physical response; full hidden-parameter mirror is undefined, so parameter projections use the preregistered left-minus-right diagnostic axis.","projection_g_abs_symmetry_axis":float(torch.dot(ga,symaxis)/ga.norm().clamp_min(1e-12)),"projection_g_abs_shared_axis":float(torch.dot(ga,shared)/ga.norm().clamp_min(1e-12)),"signed_physical_mirror_error":abs(probes["Q_SIGN_LEFT_R1"]["signed_load_balance_mean"]+probes["Q_SIGN_RIGHT_R1"]["signed_load_balance_mean"]),"yaw_p95_mirror_error":abs(probes["Q_SIGN_LEFT_R1"]["yaw_p95"]-probes["Q_SIGN_RIGHT_R1"]["yaw_p95"]),"g_abs_cosine_g_left":cos["Q_LOAD_ABS_R1"]["Q_SIGN_LEFT_R1"],"g_abs_cosine_g_right":cos["Q_LOAD_ABS_R1"]["Q_SIGN_RIGHT_R1"],"unsigned_support_side_reversals":probes["Q_LOAD_ABS_R1"]["support_side_reversal_count"],"support_yaw_sign_correlation":probes["Q_LOAD_ABS_R1"]["support_yaw_sign_correlation"]}
  result={"identity":identity,"baseline":baseline,"gradient_isolation":grad,"gradient_cosines":cos,"temporary_tensors":temps,"probes":probes,"symmetry":sym,"weights":weights,"scales":scales}
  (RAW/"worker_results.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n");wrapped.close()
 print("D21_WORKER_COMPLETE",flush=True)
if __name__=="__main__":main()
