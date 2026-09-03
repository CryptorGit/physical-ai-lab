"""A4 matched-state V2 start-boundary and bounded zero-command diagnostics."""
from __future__ import annotations
import argparse,csv,json,math,sys
from pathlib import Path
import gymnasium as gym
import torch
HERE=Path(__file__).resolve();REPO=HERE.parents[4];EXP=HERE.parent.parent
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion";OUT=BASE/"phase_w2_p1_a4_versioned_b0_label_contract_preflight";R2=BASE/"phase_w2_p1_r2_long_horizon_group_balanced_stop_integration"
SELECTED=R2/"raw/selected_student.pt";PARENT=BASE/"phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt";TEACHER=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
sys.path[:0]=[str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),str(EXP/"src"),str(HERE.parent)]
import isaaclab_tasks  # noqa:F401
import g1_omnidirectional.tasks  # noqa:F401
from g1_omnidirectional.policy import FrozenGaitActor
from g1_omnidirectional.yaw_calibration import calibrate_yaw
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
import probe_w2_p1_a4_b0_contract as a4
from train_w2_p1_student import MOVING_GROUPS,Student,load_datasets,split_groups
parser=argparse.ArgumentParser();parser.add_argument("--max-envs",type=int,default=4000);add_launcher_args(parser);args,hydra=setup_preset_cli(parser);sys.argv=[sys.argv[0],*hydra]
PROFILES=("CANDIDATE","BASE_STUDENT","PARENT_D2","PARENT_D4","N1_ALL_STOP","N2_ALL_W1B","N3_B0_STOP_B1_STOP_B2_W1B","N4_B0_STOP_B1_W1B_B2_STOP")
def minjerk(x):x=x.clamp(0,1);return x**3*(10-15*x+6*x*x)
def write_csv(path,rows):
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def clone(env,robot,term,n,p):
 origins=env.scene.env_origins;refs=torch.arange(n,device=env.device);targets=torch.arange(p*n,device=env.device);src=targets.remainder(n);local=robot.data.root_pos_w[refs]-origins[refs];pose=torch.cat((local[src]+origins[targets],robot.data.root_quat_w[refs][src]),1);vel=torch.cat((robot.data.root_lin_vel_w[refs][src],robot.data.root_ang_vel_w[refs][src]),1)
 robot.write_root_pose_to_sim(pose,targets);robot.write_root_velocity_to_sim(vel,targets);robot.write_joint_state_to_sim(robot.data.joint_pos[refs][src],robot.data.joint_vel[refs][src],env_ids=targets)
 for _,v in vars(term).items():
  if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==env.num_envs:v[targets]=v[refs][src].clone()
 env.action_manager._action[targets]=env.action_manager.action[refs][src];env.action_manager._prev_action[targets]=env.action_manager.prev_action[refs][src];env.episode_length_buf[targets]=env.episode_length_buf[refs][src];env.sim.forward()
def reconstruct(device):
 datasets,groups=load_datasets();splits=split_groups(datasets,groups);ov=torch.load(OUT/"start_boundary_b0_label_overlay_v2.pt",map_location="cpu",weights_only=False);lookup={(int(di),int(ei)):ov["target_action"][i] for i,(di,ei) in enumerate(zip(ov["dataset_index"],ov["episode_index"]))};pg=torch.Generator().manual_seed(20278210);train=splits["START_RETENTION"]["train"]
 pools={"BOUNDARY":a4.make_pool(train,datasets,"boundary",12288,pg,lookup),"START_NONBOUNDARY_V2":a4.make_pool(train,datasets,"nonboundary",8192,pg,lookup),"STOP_RECOVERY":a4.make_pool(splits["STOP_RECOVERY"]["train"],datasets,"any",8192,pg,lookup),"STEADY_STOP":a4.make_pool(splits["STEADY_STOP"]["train"],datasets,"any",8192,pg,lookup)}
 for x in MOVING_GROUPS:pools[x]=a4.make_pool(splits[x]["train"],datasets,"any",4096,pg,lookup)
 init=torch.load(SELECTED,map_location="cpu",weights_only=False)["actor_state_dict"];torch.manual_seed(20278211);gen=torch.Generator().manual_seed(20278211);m=Student(init).to(device);opt=torch.optim.Adam(m.parameters(),lr=1e-4)
 for _ in range(500):
  def loss(key,n):p=pools[key];ids=torch.randint(len(p[0]),(n,),generator=gen);o,g,t=(v[ids].to(device) for v in p);return torch.nn.functional.mse_loss(m(o,g),t)
  lb=loss("BOUNDARY",384);ls=loss("STOP_RECOVERY",256);lt=loss("STEADY_STOP",256);ln=loss("START_NONBOUNDARY_V2",256);lm=torch.stack([loss(x,64) for x in MOVING_GROUPS]).mean();total=.05*lb+.2375*(ls+lt+ln+lm);opt.zero_grad(set_to_none=True);total.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10);opt.step()
 return m.eval(),datasets,splits,lookup
def aggregate(rows):
 out=[]
 for p in PROFILES:
  for d in range(0,360,45):
   for y in (-.3,0.,.3):
    q=[r for r in rows if r["profile"]==p and r["direction"]==d and r["yaw"]==y];ats=[r["acquisition_time_s"] for r in q if r["acquisition_time_s"] is not None];out.append({"profile":p,"direction":d,"yaw":y,"episodes":len(q),"endpoint_success":sum(r["endpoint_success"] for r in q)/len(q),"acquisition_success":sum(r["acquisition_success"] for r in q)/len(q),"fall_rate":sum(r["fall"] for r in q)/len(q),"dangerous_slip_rate":sum(r["dangerous_slip"] for r in q)/len(q),"impact_rate":sum(r["impact"] for r in q)/len(q),"acquisition_median_s":float(torch.tensor(ats).median()) if ats else None,"acquisition_p95_s":float(torch.tensor(ats).quantile(.95)) if ats else None})
 return out
def main():
 jobs=[(d,y,e) for d in range(0,360,45) for y in (-.3,0.,.3) for e in range(200)];p=len(PROFILES);cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=min(args.max_envs,p*len(jobs));cfg.episode_length_s=12.;cfg.seed=20278221;cfg.observations.policy.enable_corruption=False
 if args.device:cfg.sim.device=agent.device=args.device
 rows=[];traces=[]
 with launch_simulation(cfg,args):
  w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);env=w.unwrapped;dev=env.device;candidate,_,_,_=reconstruct(dev);base=Student(torch.load(SELECTED,map_location="cpu",weights_only=False)["actor_state_dict"]).to(dev).eval();parent=FrozenGaitActor(PARENT).to(dev).eval();teacher=FrozenGaitActor(TEACHER).to(dev).eval();robot=env.scene["robot"];sensor=env.scene.sensors["contact_forces"];feet=sensor.find_bodies(".*_ankle_roll_link")[0];rfeet=robot.find_bodies(".*_ankle_roll_link")[0];term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True;nmax=env.num_envs//p;cursor=0;gait=torch.zeros(env.num_envs,device=dev)
  while cursor<len(jobs):
   batch=jobs[cursor:cursor+nmax];n=len(batch);pad=batch+[batch[i%n] for i in range(nmax-n)] if n<nmax else batch;ids=torch.arange(p*nmax,device=dev);env.reset(env_ids=ids);term.external_override.zero_();term._update_command();obs=w.get_observations().to(dev)
   for _ in range(round(3/env.step_dt)):
    with torch.inference_mode():a=teacher(obs["policy"],gait)
    obs,_,_,_=w.step(a);obs=obs.to(dev)
   clone(env,robot,term,nmax,p);term.external_override.zero_();term._update_command();obs=w.get_observations().to(dev)
   with torch.inference_mode():ca=candidate(obs["policy"],gait);ba=base(obs["policy"],gait);pa=parent(obs["policy"],gait);ta=teacher(obs["policy"],gait)
   act=ca.clone()
   for bi,name in enumerate(PROFILES):
    sl=slice(bi*nmax,(bi+1)*nmax)
    if name=="BASE_STUDENT":act[sl]=ba[sl]
    elif name in ("PARENT_D2","PARENT_D4","N2_ALL_W1B"):act[sl]=pa[sl]
    elif name.startswith("N1_") or name.startswith("N3_") or name.startswith("N4_"):act[sl]=ta[sl]
   dirs=torch.tensor([math.radians(x[0]) for x in pad],device=dev);yb=torch.tensor([x[1] for x in pad],device=dev);target=torch.stack((.3*dirs.cos(),.3*dirs.sin(),yb),1).repeat(p,1);obs,_,done,ex=w.step(act);obs=obs.to(dev);fall=done.bool()&~ex.get("time_outs",torch.zeros_like(done)).bool();slip=torch.zeros_like(fall);impact=torch.zeros_like(fall);streak=torch.zeros(env.num_envs,dtype=torch.long,device=dev);acq=torch.full((env.num_envs,),float("nan"),device=dev);sustain=torch.zeros_like(streak);ve_sum=torch.zeros_like(acq);ye_sum=torch.zeros_like(acq);measure=torch.zeros_like(acq)
   for k in range(1,round(5.5/env.step_dt)+1):
    t=k*env.step_dt;alpha=minjerk(torch.tensor(t/1.5,device=dev));physical=target*alpha;term.external_override[:,:2]=physical[:,:2];term.external_override[:,2]=calibrate_yaw(physical[:,2]);term._update_command();obs=w.get_observations().to(dev)
    with torch.inference_mode():ca=candidate(obs["policy"],gait);ba=base(obs["policy"],gait);pa=parent(obs["policy"],gait);ta=teacher(obs["policy"],gait)
    a=ca.clone()
    for bi,name in enumerate(PROFILES):
     sl=slice(bi*nmax,(bi+1)*nmax)
     if name=="BASE_STUDENT":a[sl]=ba[sl]
     elif name=="PARENT_D2":a[sl]=pa[sl] if k<2 else ba[sl]
     elif name=="PARENT_D4":a[sl]=pa[sl] if k<4 else ba[sl]
     elif name=="N1_ALL_STOP":a[sl]=ta[sl] if k<=2 else ba[sl]
     elif name=="N2_ALL_W1B":a[sl]=pa[sl] if k<=2 else ba[sl]
     elif name=="N3_B0_STOP_B1_STOP_B2_W1B":a[sl]=ta[sl] if k==1 else pa[sl] if k==2 else ba[sl]
     elif name=="N4_B0_STOP_B1_W1B_B2_STOP":a[sl]=pa[sl] if k==1 else ta[sl] if k==2 else ba[sl]
    obs,_,dn,x=w.step(a);obs=obs.to(dev);fall|=dn.bool()&~x.get("time_outs",torch.zeros_like(dn)).bool();force=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);fs=torch.linalg.vector_norm(robot.data.body_lin_vel_w[:,rfeet,:2],dim=-1);bad=(fs>.55)&(force>5);streak=torch.where(bad.any(1),streak+1,torch.zeros_like(streak));slip|=streak>=5;impact|=force.amax(1)>3500;actual=robot.data.root_lin_vel_b[:,:2];ay=robot.data.root_ang_vel_b[:,2];ve=torch.linalg.vector_norm(actual-target[:,:2],dim=1);ye=(ay-target[:,2]).abs();sign=(target[:,2].abs()<1e-8)|(ay*target[:,2]>0);ep=(ve<=.25)&(ye<=.20)&sign&~fall&~slip&~impact
    if t>=1.5:sustain=torch.where(ep,sustain+1,torch.zeros_like(sustain));new=torch.isnan(acq)&(sustain>=round(.2/env.step_dt));acq[new]=t-1.5
    if t>=3.5:ve_sum+=ve;ye_sum+=ye;measure+=1
    if k in (1,2,4,8,16):
     for bi,name in enumerate(PROFILES[:4]):
      sl=slice(bi*nmax,(bi+1)*nmax);traces.append({"profile":name,"batch_start":cursor,"steps":k,"mean_linear_speed":float(torch.linalg.vector_norm(robot.data.root_lin_vel_b[sl],dim=1).mean()),"mean_angular_speed":float(torch.linalg.vector_norm(robot.data.root_ang_vel_b[sl],dim=1).mean()),"mean_roll":float(robot.data.projected_gravity_b[sl,1].mean()),"mean_pitch":float(robot.data.projected_gravity_b[sl,0].mean()),"fall_rate":float(fall[sl].float().mean()),"slip_rate":float(slip[sl].float().mean())})
   endpoint=(ve_sum/measure<=.25)&(ye_sum/measure<=.20)&~fall&~slip&~impact
   for bi,name in enumerate(PROFILES):
    for i,(d,y,e) in enumerate(pad[:n]):
     j=bi*nmax+i;rows.append({"profile":name,"direction":d,"yaw":y,"episode":e,"endpoint_success":int(endpoint[j]),"acquisition_success":int(torch.isfinite(acq[j]) and acq[j]<=3),"acquisition_time_s":None if torch.isnan(acq[j]) else float(acq[j]),"fall":int(fall[j]),"dangerous_slip":int(slip[j]),"impact":int(impact[j])})
   cursor+=n;print(json.dumps({"matched_trials":cursor,"total":len(jobs)}),flush=True)
  summary=aggregate(rows);cand=[x for x in summary if x["profile"]=="CANDIDATE"];ref=[x for x in summary if x["profile"]=="PARENT_D2"];agg=lambda q,k:sum(x[k]*x["episodes"] for x in q)/sum(x["episodes"] for x in q);ce,ca,cf,cs,ci=(agg(cand,k) for k in ("endpoint_success","acquisition_success","fall_rate","dangerous_slip_rate","impact_rate"));re,ra=agg(ref,"endpoint_success"),agg(ref,"acquisition_success");worste=max(abs(x["endpoint_success"]-y["endpoint_success"]) for x,y in zip(cand,ref));worsta=max(abs(x["acquisition_success"]-y["acquisition_success"]) for x,y in zip(cand,ref));passed=ce>=re-.05 and ca>=ra-.05 and cf<=.05 and cs<=.10 and ci<=.05 and worste<=.10 and worsta<=.10
  write_csv(OUT/"validation_v2_physical_start_gate.csv",summary);(OUT/"validation_v2_physical_start_gate.json").write_text(json.dumps({"pass":passed,"candidate":{"endpoint":ce,"acquisition":ca,"fall":cf,"dangerous_slip":cs,"impact":ci},"reference":{"endpoint":re,"acquisition":ra},"differences":{"aggregate_endpoint_pp":100*(ce-re),"aggregate_acquisition_pp":100*(ca-ra),"worst_condition_endpoint_pp":100*worste,"worst_condition_acquisition_pp":100*worsta},"conditions":cand},indent=2)+"\n");write_csv(OUT/"v2_start_boundary_trajectory_analysis.csv",traces);(OUT/"v2_start_boundary_trajectory_analysis.json").write_text(json.dumps({"traces":traces,"interpretation":"B0 candidate followed by candidate-only B1+; matched positive controls use bounded diagnostic intervention only."},indent=2)+"\n");(OUT/"v2_boundary_sequence_negative_controls.json").write_text(json.dumps({"profiles":[x for x in summary if x["profile"].startswith("N")],"candidate_selection_use":False},indent=2)+"\n");w.close()
if __name__=="__main__":main()
