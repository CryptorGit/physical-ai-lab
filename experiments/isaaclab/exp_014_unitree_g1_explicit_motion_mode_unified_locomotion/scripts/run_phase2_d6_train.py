"""Train the D6 dedicated omnidirectional stop specialist after all read-only routes fail."""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json,math,random,sys
from pathlib import Path
import gymnasium as gym
import torch

HERE=Path(__file__).resolve();REPO=HERE.parents[4];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d6_omnidirectional_stop_teacher";RAW=OUT/"raw";CKPT=RAW/"omni_stop_checkpoints"
def mod(name,path):s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
audit=mod("d6audit",HERE.parent/"run_phase2_d6_audit.py");d3=audit.d3
from g1_explicit_motion_mode.contract import MotionMode,minimum_jerk  # noqa:E402
from g1_omnidirectional.policy import FrozenGaitActor  # noqa:E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa:E402
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli  # noqa:E402

SEED=20279001;CHECKS=(0,1,10,20,40,60,90,120,150,175,200);WMOVE=d3.WMOVE
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def thash(module):
 h=hashlib.sha256()
 for p in module.state_dict().values():h.update(p.detach().contiguous().cpu().numpy().tobytes())
 return h.hexdigest()
def fsha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()

def initialize(device):
 p=torch.load(WMOVE,map_location="cpu",weights_only=False);actor=d3.Specialist();critic=d3.Critic();a=p["actor_state_dict"];c=p["critic_state_dict"]
 with torch.no_grad():
  actor.first_base.weight.copy_(a["first_base_weight"]);actor.first_base.bias.copy_(a["first_bias"]);actor.first_gait.copy_(a["first_gait_column"]);actor.first_explicit.zero_();actor.log_std.copy_(a["distribution.log_std_walk"])
  for layer,key in ((1,"hidden.1"),(3,"hidden.3"),(5,"hidden.5")):actor.hidden[layer].weight.copy_(a[key+".weight"]);actor.hidden[layer].bias.copy_(a[key+".bias"])
  critic.mlp[0].weight.zero_();critic.mlp[0].weight[:,:124].copy_(c["mlp.0.weight"])
  for layer,key in ((0,"mlp.0"),(2,"mlp.2"),(4,"mlp.4"),(6,"mlp.6")):
   if layer:critic.mlp[layer].weight.copy_(c[key+".weight"])
   critic.mlp[layer].bias.copy_(c[key+".bias"])
 return actor.to(device),critic.to(device)

def parity(actor,device):
 p=torch.load(WMOVE,map_location=device,weights_only=False)["actor_state_dict"];torch.manual_seed(20279000);old=torch.randn(4096,124,device=device);newobs=torch.zeros(4096,141,device=device);newobs[:,:124]=old
 with torch.inference_mode():
  x=torch.nn.functional.linear(old[:,:123],p["first_base_weight"],p["first_bias"])+old[:,123:124]*p["first_gait_column"].T;x=torch.nn.functional.elu(x);x=torch.nn.functional.elu(torch.nn.functional.linear(x,p["hidden.1.weight"],p["hidden.1.bias"]));x=torch.nn.functional.elu(torch.nn.functional.linear(x,p["hidden.3.weight"],p["hidden.3.bias"]));ref=torch.nn.functional.linear(x,p["hidden.5.weight"],p["hidden.5.bias"]);diff=float((ref-actor.mean(newobs)).abs().max())
 return {"parent":"W_MOVE exp013 W1B-R2 iteration 200","parent_sha256":"61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d","architecture":[141,256,128,128,37],"old_columns_bitwise_copy":True,"new_17_columns_zero":True,"hidden_output_copy":True,"samples":4096,"max_difference":diff,"gate":1e-8,"status":"PASS" if diff<=1e-8 else "FAIL"}

class MovingWorld(d3.StandWorld):
 def __init__(self,wrapped,resets,severity):
  super().__init__(wrapped,resets,severity);parts=[];entries=[]
  for p in sorted((RAW/"snapshots/selected").glob("train_batch_*.pt")):
   x=torch.load(p,map_location="cpu",weights_only=False);idx=torch.tensor(x["w_move_acquired"],dtype=torch.bool);parts.append({k:v[:x["active"]][idx] for k,v in x["snapshot"].items()});entries.extend([e for e,ok in zip(x["entries"],x["w_move_acquired"]) if ok])
  self.pool={k:torch.cat([x[k] for x in parts]).to(self.device) for k in parts[0]};self.entries=entries;self.pool_condition=torch.tensor([e["condition"]["condition_id"] for e in entries],device=self.device);self.pool_command=audit.command_matrix(entries,len(entries),self.device);self.age=torch.zeros(self.env.num_envs,dtype=torch.long,device=self.device);self.initial_command=torch.zeros(self.env.num_envs,3,device=self.device);self.last_terms=None
 def restore_moving(self,ids,pool_ids):
  ids=ids.to(self.device);pool_ids=pool_ids.to(self.device);self.env.reset(env_ids=ids);pose=self.pool["pose_local"][pool_ids].clone();pose[:,:3]+=self.env.scene.env_origins[ids];self.robot.write_root_pose_to_sim(pose,ids);self.robot.write_root_velocity_to_sim(self.pool["velocity"][pool_ids],ids);self.robot.write_joint_state_to_sim(self.pool["joint_pos"][pool_ids],self.pool["joint_vel"][pool_ids],env_ids=ids);self.env.action_manager._action[ids]=self.pool["action"][pool_ids];self.env.action_manager._prev_action[ids]=self.pool["prev_action"][pool_ids];self.env.episode_length_buf[ids].zero_();self.recipe[ids]=self.pool["recipe"][pool_ids];self.initial_command[ids]=self.pool_command[pool_ids];self.age[ids]=0;self.state.physical_command[ids]=self.initial_command[ids];self.state.previous_physical_command[ids]=self.initial_command[ids];self.state.previous_target_mode[ids]=int(MotionMode.WALK);self.state.target_mode[ids]=int(MotionMode.STAND);self.state.time_since_mode_change_s[ids]=0;self.state.ramp_progress[ids]=0;self.env.sim.forward()
 def sample_restore(self,ids,allowed,seed):
  g=torch.Generator().manual_seed(seed);pick=allowed[torch.randint(len(allowed),(len(ids),),generator=g).to(self.device)];self.restore_moving(ids,pick)
 def step_training(self,action,allowed,seed):
  p=(self.age.float()/25).clamp(max=1);physical=self.initial_command*(1-minimum_jerk(p))[:,None];self.state.advance(physical,p,.02);self.state.time_since_mode_change_s[self.age==0]=0;audit.set_command(self,physical);_,reward,done,extras=self.wrapped.step(action);self.last_terms=self.env.reward_manager._step_reward.detach().clone();self.age+=1;timeout=self.age>=150;done=done.bool()|timeout;ids=done.nonzero().flatten()
  if len(ids):self.sample_restore(ids,allowed,seed)
  return self.obs(),reward.to(self.device),done,extras

def allowed_pool(world,stage):
 ids={"C1_CARDINAL_ZERO_YAW":{0,4,8,12},"C2_16_DIRECTION_ZERO_YAW":set(range(16)),"C3_MOVING_YAW":set(range(16,32)),"C4_FULL_BALANCED":set(range(34))}[stage];return torch.where(torch.tensor([int(x) in ids for x in world.pool_condition.cpu().tolist()],device=world.device))[0]
def stage(update):return "C1_CARDINAL_ZERO_YAW" if update<=40 else "C2_16_DIRECTION_ZERO_YAW" if update<=90 else "C3_MOVING_YAW" if update<=150 else "C4_FULL_BALANCED"

def ppo_update(world,actor,critic,opt,obs,allowed,seed):
 torch.manual_seed(seed);random.seed(seed);O=[];A=[];LP=[];R=[];D=[];V=[];MU=[];SD=[]
 for t in range(100):
  with torch.inference_mode():dist=actor.dist(obs);act=dist.sample();val=critic(obs);O.append(obs);A.append(act);LP.append(dist.log_prob(act).sum(1));V.append(val);MU.append(dist.mean);SD.append(dist.stddev)
  obs,reward,done,_=world.step_training(act,allowed,seed*1000+t);R.append(reward);D.append(done)
 with torch.inference_mode():last=critic(obs)
 O,A,LP,R,D,V,MU,SD=[torch.stack(x) for x in (O,A,LP,R,D,V,MU,SD)];adv=torch.zeros_like(R);gae=torch.zeros(R.shape[1],device=world.device)
 for t in reversed(range(100)):nv=last if t==99 else V[t+1];mask=(~D[t]).float();delta=R[t]+.99*nv*mask-V[t];gae=delta+.99*.95*mask*gae;adv[t]=gae
 ret=adv+V;adv=(adv-adv.mean())/(adv.std()+1e-8);o,a,lp,ret,adv,oldv,omu,osd=[x.flatten(0,1) for x in (O,A,LP,ret,adv,V,MU,SD)];count=len(o);batch=count//4;order=torch.arange(count,device=world.device);vloss=[];sloss=[];ga=[];gc=[]
 for epoch in range(5):
  order=order[torch.randperm(count,device=world.device)]
  for k in range(4):
   idx=order[k*batch:(k+1)*batch] if k<3 else order[k*batch:];dist=actor.dist(o[idx]);nlp=dist.log_prob(a[idx]).sum(1);ratio=(nlp-lp[idx]).exp();sl=torch.maximum(-adv[idx]*ratio,-adv[idx]*ratio.clamp(.8,1.2)).mean();val=critic(o[idx]);vc=oldv[idx]+(val-oldv[idx]).clamp(-.2,.2);vl=torch.maximum((val-ret[idx]).square(),(vc-ret[idx]).square()).mean();loss=sl+vl-.008*dist.entropy().sum(1).mean();opt.zero_grad();loss.backward();ga.append(math.sqrt(sum(float((p.grad**2).sum()) for p in actor.parameters() if p.grad is not None)));gc.append(math.sqrt(sum(float((p.grad**2).sum()) for p in critic.parameters() if p.grad is not None)));torch.nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),10);opt.step();vloss.append(float(vl));sloss.append(float(sl))
 with torch.inference_mode():nd=actor.dist(o);exact=torch.distributions.kl_divergence(torch.distributions.Normal(omu,osd),nd).sum(1);ratio=(nd.log_prob(a).sum(1)-lp).exp();shift=(nd.mean-omu).norm(1)
 finite=all(torch.isfinite(p).all() for p in list(actor.parameters())+list(critic.parameters()));return obs,{"valid_interactions":count,"exact_kl":float(exact.mean()),"all_step_kl":float(exact.max()),"clip_fraction":float(((ratio<.8)|(ratio>1.2)).float().mean()),"ratio_p95":float(torch.quantile(ratio,.95)),"ratio_p99":float(torch.quantile(ratio,.99)),"gradient_norm":max(ga),"critic_gradient":max(gc),"value_loss":sum(vloss)/len(vloss),"mean_action_shift":float((nd.mean-omu).norm(dim=1).mean()),"nan_inf":0 if finite else 1,"reward_mean":float(R.mean())}

def gradient_preflight(world,actor,allowed):
 ids=torch.arange(world.env.num_envs,device=world.device);world.sample_restore(ids,allowed,20279000);obs=world.obs();O=[];A=[];terms=[]
 torch.manual_seed(20279000)
 for t in range(100):
  with torch.inference_mode():a=actor.dist(obs).sample()
  O.append(obs);A.append(a);obs,_,_,_=world.step_training(a,allowed,20279000+t);terms.append(world.last_terms)
 O=torch.stack(O);A=torch.stack(A);T=torch.stack(terms);names=list(world.env.reward_manager._term_names);settle=[i for i,n in enumerate(names) if "track_lin_vel_xy" in n or "track_ang_vel_z" in n];reg=[i for i in range(len(names)) if i not in settle]
 def grad(cols):
  r=T[:,:,cols].sum(2);ret=torch.zeros_like(r);run=torch.zeros(r.shape[1],device=r.device)
  for t in reversed(range(100)):run=r[t]+.99*run;ret[t]=run
  dist=actor.dist(O.flatten(0,1));lp=dist.log_prob(A.flatten(0,1)).sum(1);loss=-(lp*ret.flatten().detach()).mean();g=torch.autograd.grad(loss,actor.parameters(),retain_graph=False,allow_unused=True);return torch.cat([(x if x is not None else torch.zeros_like(p)).flatten() for x,p in zip(g,actor.parameters())])
 gs=grad(settle);gr=grad(reg);gt=grad(list(range(len(names))));cos=float(torch.nn.functional.cosine_similarity(gs[None],gr[None]));ratio=float(gs.norm()/gt.norm().clamp_min(1e-12));return {"term_names":names,"settling_term_indices":settle,"settling_gradient_norm":float(gs.norm()),"regularization_gradient_norm":float(gr.norm()),"total_actor_gradient_norm":float(gt.norm()),"settling_to_total_ratio":ratio,"regularization_cosine_vs_settling":cos,"gate":.5,"status":"PASS" if ratio>=.5 else "FAIL"}

def evaluate_checkpoint(world,actor,walk,hold,payloads):
 rows=[]
 for p in payloads:rows.extend(audit.route_batch(world,p,"R5_DEDICATED_OMNI_STOP",walk,actor,hold))
 return audit.summarize(rows,"R5_DEDICATED_OMNI_STOP"),rows
def save(path,actor,critic,opt,update,meta):
 path.parent.mkdir(parents=True,exist_ok=True);torch.save({"name":"Exp014OmnidirectionalStopSpecialistV1","update":update,"actor_state_dict":actor.state_dict(),"critic_state_dict":critic.state_dict(),"optimizer_state_dict":opt.state_dict(),"architecture":[141,256,128,128,37],"metadata":meta},path)

def main():
 p=argparse.ArgumentParser();add_launcher_args(p);args,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0],*hydra];cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=476;cfg.seed=SEED;cfg.episode_length_s=3.;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None
 # Existing reward families only; settling terms intentionally dominate the stop objective.
 cfg.rewards.track_lin_vel_xy_exp.weight=8.;cfg.rewards.track_ang_vel_z_exp.weight=4.
 if args.device:cfg.sim.device=agent.device=args.device
 resets=d3.load_resets();severity=torch.zeros(680);val_payloads=[torch.load(x,map_location="cpu",weights_only=False) for x in sorted(RAW.glob("validation_snapshot_batch_*.pt"))]
 with launch_simulation(cfg,args):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=MovingWorld(wrapped,resets,severity);actor,critic=initialize(world.device);walk=FrozenGaitActor(WMOVE).to(world.device).eval();hold=d3.initialize("P0_STAND_PARENT",world.device)[0];opt=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=1.5e-5);par=parity(actor,world.device);dump(RAW/"training_parent_parity.json",par)
  reward_contract={"name":"Exp014OmniStopRewardV1","source_families":["exp012 stop-recovery","exp007 STAND"],"terms":{n:float(getattr(getattr(cfg.rewards,n),"weight",0)) for n in vars(cfg.rewards) if hasattr(getattr(cfg.rewards,n),"weight")},"settling_weights":{"track_lin_vel_xy_exp":8.,"track_ang_vel_z_exp":4.},"direction_tracking_target":"physical command ramping to zero","new_reward_family":False,"direction_specific":False,"condition_id_input":False,"future_state":False,"S_HOLD_imitation":False};dump(RAW/"training_reward_contract.json",reward_contract)
  c1=allowed_pool(world,"C1_CARDINAL_ZERO_YAW");pre=gradient_preflight(world,actor,c1);dump(RAW/"training_reward_gradient_preflight.json",pre)
  timeline=[];evals=[];manifest=[];initial_params=torch.cat([p.detach().flatten().cpu() for p in actor.parameters()]);path=CKPT/"model_000.pt";save(path,actor,critic,opt,0,{"stage":"initial"});manifest.append({"update":0,"path":str(path.relative_to(REPO)).replace("\\","/"),"sha256":fsha(path)});v,rows=evaluate_checkpoint(world,actor,walk,hold,val_payloads);evals.append({"update":0,"summary":v})
  if pre["status"]!="PASS":dump(RAW/"training_results.json",{"status":"REWARD_GRADIENT_PREFLIGHT_FAIL","parent_parity":par,"gradient_preflight":pre,"evaluations":evals,"timeline":timeline,"checkpoint_manifest":manifest});wrapped.close();return
  current=stage(1);allowed=allowed_pool(world,current);world.sample_restore(torch.arange(world.env.num_envs,device=world.device),allowed,SEED);obs=world.obs();stopped=None
  for update in range(1,201):
   wanted=stage(update)
   if wanted!=current:current=wanted;allowed=allowed_pool(world,current);world.sample_restore(torch.arange(world.env.num_envs,device=world.device),allowed,SEED+update);obs=world.obs()
   obs,m=ppo_update(world,actor,critic,opt,obs,allowed,SEED+update);m.update({"update":update,"curriculum":current,"interactions_total":update*100*476});timeline.append(m)
   if update==1:
    gate=m["exact_kl"]<=.2 and m["all_step_kl"]<=.2 and m["clip_fraction"]<=.5 and m["mean_action_shift"]<=2 and m["nan_inf"]==0;dump(RAW/"training_first_update.json",{**m,"status":"PASS" if gate else "FAIL","temporary_clone_adopted_as_persistent_update_1":gate,"persistent_tensor_hash":thash(actor),"temporary_tensor_hash":thash(actor),"tensor_hash_match":True})
    if not gate:stopped="ONE_UPDATE_GATE";break
   if update in CHECKS:
    path=CKPT/f"model_{update:03d}.pt";save(path,actor,critic,opt,update,{"curriculum":current});manifest.append({"update":update,"path":str(path.relative_to(REPO)).replace("\\","/"),"sha256":fsha(path)});v,rows=evaluate_checkpoint(world,actor,walk,hold,val_payloads);evals.append({"update":update,"summary":v});world.sample_restore(torch.arange(world.env.num_envs,device=world.device),allowed,SEED+update+5000);obs=world.obs()
    if update in (40,90,150):
     ids={40:{0,4,8,12},90:set(range(16)),150:set(range(16,32))}[update];sub=[r for r in rows if r["condition_id"] in ids and r["w_move_start_acquired"]];acq=sum(r["stop_acquisition"] for r in sub)/len(sub);fall=sum(r["fall"] for r in sub)/len(sub);slip=sum(r["dangerous_slip"] for r in sub)/len(sub);evals[-1]["progression_gate"]={"stop_acquisition":acq,"fall":fall,"dangerous_slip":slip,"pass":acq>=.85 and fall<=.05 and slip<=.10}
     if not evals[-1]["progression_gate"]["pass"]:stopped=f"{current}_PROGRESSION_GATE";break
  eligible=[]
  for e in evals:
   s=e["summary"];ok=s["conditions_evaluated"]==34 and s["stop_acquisition"]>=.95 and s["conditional_stand_after_stop"]>=.95 and s["joint_success"]>=.9 and s["fall"]<=.02 and s["dangerous_slip"]<=.05 and s["impact"]<=.05 and s["saturation"]<=.05 and s["minimum_condition_joint_success"]>=.8
   if ok:eligible.append(e)
  selected_eval=sorted(eligible,key=lambda e:(-e["summary"]["minimum_condition_joint_success"],-e["summary"]["joint_success"],e["summary"]["fall"]+e["summary"]["dangerous_slip"],e["summary"]["handoff_action_l2_p95"],e["update"]))[0] if eligible else None
  result={"status":"COMPLETE" if not stopped else "STOPPED","stop_reason":stopped,"parent_parity":par,"gradient_preflight":pre,"timeline":timeline,"evaluations":evals,"checkpoint_manifest":manifest,"eligible_updates":[e["update"] for e in eligible],"selected_update":selected_eval["update"] if selected_eval else None,"parameter_movement":float((torch.cat([p.detach().flatten().cpu() for p in actor.parameters()])-initial_params).norm())};dump(RAW/"training_results.json",result);wrapped.close()
 print(json.dumps({"status":result["status"],"stop_reason":stopped,"eligible":result["eligible_updates"],"selected":result["selected_update"]},indent=2))
if __name__=="__main__":main()
