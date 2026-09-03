"""A5 four-step positive control and candidate-visited V3 overlay collection."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
import gymnasium as gym
import torch
HERE=Path(__file__).resolve();REPO=HERE.parents[4];EXP=HERE.parent.parent
sys.path[:0]=[str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),str(EXP/"src"),str(HERE.parent)]
import isaaclab_tasks  # noqa:F401
import g1_omnidirectional.tasks  # noqa:F401
from g1_omnidirectional.policy import FrozenGaitActor
from g1_omnidirectional.yaw_calibration import calibrate_yaw
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
import w2_p1_a5_common as c
OUT=c.A5;BASE=c.BASE;PARENT=BASE/"phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt";TEACHER=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
parser=argparse.ArgumentParser();parser.add_argument("--max-envs",type=int,default=4000);add_launcher_args(parser);args,hydra=setup_preset_cli(parser);sys.argv=[sys.argv[0],*hydra]
PROFILES=("PC0_CANDIDATE_ONLY","PC1_B0_STOP_B1_B2_W1B","PC2_B0_STOP_B1_B4_W1B","PC3_B0_B4_W1B")
def minjerk(x):x=x.clamp(0,1);return x**3*(10-15*x+6*x*x)
def write_csv(path,rows):
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def clone(env,robot,term,n,p):
 origins=env.scene.env_origins;refs=torch.arange(n,device=env.device);targets=torch.arange(p*n,device=env.device);src=targets.remainder(n);local=robot.data.root_pos_w[refs]-origins[refs];pose=torch.cat((local[src]+origins[targets],robot.data.root_quat_w[refs][src]),1);vel=torch.cat((robot.data.root_lin_vel_w[refs][src],robot.data.root_ang_vel_w[refs][src]),1);robot.write_root_pose_to_sim(pose,targets);robot.write_root_velocity_to_sim(vel,targets);robot.write_joint_state_to_sim(robot.data.joint_pos[refs][src],robot.data.joint_vel[refs][src],env_ids=targets)
 for _,v in vars(term).items():
  if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==env.num_envs:v[targets]=v[refs][src].clone()
 env.action_manager._action[targets]=env.action_manager.action[refs][src];env.action_manager._prev_action[targets]=env.action_manager.prev_action[refs][src];env.episode_length_buf[targets]=env.episode_length_buf[refs][src];env.sim.forward()
def summarize(rows):
 out=[]
 for p in PROFILES:
  for d in range(0,360,45):
   for y in (-.3,0.,.3):
    q=[r for r in rows if r["profile"]==p and r["direction"]==d and r["yaw"]==y];ats=[r["acquisition_time_s"] for r in q if r["acquisition_time_s"] is not None];out.append({"profile":p,"direction":d,"yaw":y,"episodes":len(q),"endpoint_success":sum(r["endpoint_success"] for r in q)/len(q),"acquisition_success":sum(r["acquisition_success"] for r in q)/len(q),"fall_rate":sum(r["fall"] for r in q)/len(q),"dangerous_slip_rate":sum(r["dangerous_slip"] for r in q)/len(q),"impact_rate":sum(r["impact"] for r in q)/len(q),"acquisition_median_s":float(torch.tensor(ats).median()) if ats else None,"acquisition_p95_s":float(torch.tensor(ats).quantile(.95)) if ats else None})
 return out
def positive(env,w,robot,sensor,term,candidate,parent,teacher,gait):
 jobs=[(d,y,e) for d in range(0,360,45) for y in (-.3,0.,.3) for e in range(200)];p=4;nmax=env.num_envs//p;rows=[];cursor=0;feet=sensor.find_bodies(".*_ankle_roll_link")[0];rfeet=robot.find_bodies(".*_ankle_roll_link")[0]
 while cursor<len(jobs):
  b=jobs[cursor:cursor+nmax];n=len(b);pad=b+[b[i%n] for i in range(nmax-n)] if n<nmax else b;ids=torch.arange(p*nmax,device=env.device);env.reset(env_ids=ids);term.external_override.zero_();term._update_command();obs=w.get_observations().to(env.device)
  for _ in range(round(3/env.step_dt)):
   with torch.inference_mode():a=teacher(obs["policy"],gait)
   obs,_,_,_=w.step(a);obs=obs.to(env.device)
  clone(env,robot,term,nmax,p);term.external_override.zero_();term._update_command();obs=w.get_observations().to(env.device)
  with torch.inference_mode():ca=candidate(obs["policy"],gait);pa=parent(obs["policy"],gait);ta=teacher(obs["policy"],gait)
  a=ca.clone();a[nmax:3*nmax]=ta[nmax:3*nmax];a[3*nmax:]=pa[3*nmax:];dirs=torch.tensor([math.radians(x[0]) for x in pad],device=env.device);yb=torch.tensor([x[1] for x in pad],device=env.device);target=torch.stack((.3*dirs.cos(),.3*dirs.sin(),yb),1).repeat(p,1);obs,_,dn,x=w.step(a);obs=obs.to(env.device);fall=dn.bool()&~x.get("time_outs",torch.zeros_like(dn)).bool();slip=torch.zeros_like(fall);impact=torch.zeros_like(fall);streak=torch.zeros(env.num_envs,dtype=torch.long,device=env.device);acq=torch.full((env.num_envs,),float("nan"),device=env.device);sustain=torch.zeros_like(streak);ve_sum=torch.zeros_like(acq);ye_sum=torch.zeros_like(acq);measure=torch.zeros_like(acq)
  for k in range(1,round(5.5/env.step_dt)+1):
   t=k*env.step_dt;alpha=minjerk(torch.tensor(t/1.5,device=env.device));physical=target*alpha;term.external_override[:,:2]=physical[:,:2];term.external_override[:,2]=calibrate_yaw(physical[:,2]);term._update_command();obs=w.get_observations().to(env.device)
   with torch.inference_mode():ca=candidate(obs["policy"],gait);pa=parent(obs["policy"],gait)
   a=ca.clone();a[nmax:2*nmax]=pa[nmax:2*nmax] if k<=2 else ca[nmax:2*nmax];a[2*nmax:3*nmax]=pa[2*nmax:3*nmax] if k<=4 else ca[2*nmax:3*nmax];a[3*nmax:]=pa[3*nmax:] if k<=4 else ca[3*nmax:]
   obs,_,dn,x=w.step(a);obs=obs.to(env.device);fall|=dn.bool()&~x.get("time_outs",torch.zeros_like(dn)).bool();force=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);fs=torch.linalg.vector_norm(robot.data.body_lin_vel_w[:,rfeet,:2],dim=-1);bad=(fs>.55)&(force>5);streak=torch.where(bad.any(1),streak+1,torch.zeros_like(streak));slip|=streak>=5;impact|=force.amax(1)>3500;actual=robot.data.root_lin_vel_b[:,:2];ay=robot.data.root_ang_vel_b[:,2];ve=torch.linalg.vector_norm(actual-target[:,:2],dim=1);ye=(ay-target[:,2]).abs();sign=(target[:,2].abs()<1e-8)|(ay*target[:,2]>0);ep=(ve<=.25)&(ye<=.20)&sign&~fall&~slip&~impact
   if t>=1.5:sustain=torch.where(ep,sustain+1,torch.zeros_like(sustain));new=torch.isnan(acq)&(sustain>=round(.2/env.step_dt));acq[new]=t-1.5
   if t>=3.5:ve_sum+=ve;ye_sum+=ye;measure+=1
  endpoint=(ve_sum/measure<=.25)&(ye_sum/measure<=.20)&~fall&~slip&~impact
  for bi,pn in enumerate(PROFILES):
   for i,(d,y,e) in enumerate(pad[:n]):
    j=bi*nmax+i;rows.append({"profile":pn,"direction":d,"yaw":y,"episode":e,"endpoint_success":int(endpoint[j]),"acquisition_success":int(torch.isfinite(acq[j]) and acq[j]<=3),"acquisition_time_s":None if torch.isnan(acq[j]) else float(acq[j]),"fall":int(fall[j]),"dangerous_slip":int(slip[j]),"impact":int(impact[j])})
  cursor+=n;print(json.dumps({"positive_control":cursor,"total":len(jobs)}),flush=True)
 summary=summarize(rows);pc2=[x for x in summary if x["profile"]=="PC2_B0_STOP_B1_B4_W1B"];n=sum(x["episodes"] for x in pc2);av=lambda k:sum(x[k]*x["episodes"] for x in pc2)/n;gate=av("endpoint_success")>=.95 and av("acquisition_success")>=.90 and av("fall_rate")<=.02 and av("dangerous_slip_rate")<=.10 and av("impact_rate")<=.05 and min(x["endpoint_success"] for x in pc2)>=.90 and min(x["acquisition_success"] for x in pc2)>=.85
 write_csv(OUT/"four_step_runtime_positive_control.csv",summary);(OUT/"four_step_runtime_positive_control.json").write_text(json.dumps({"profiles":summary,"PC2_gate":{"pass":gate,"aggregate_endpoint":av("endpoint_success"),"aggregate_acquisition":av("acquisition_success"),"fall":av("fall_rate"),"dangerous_slip":av("dangerous_slip_rate"),"impact":av("impact_rate"),"minimum_condition_endpoint":min(x["endpoint_success"] for x in pc2),"minimum_condition_acquisition":min(x["acquisition_success"] for x in pc2)}},indent=2)+"\n");return gate
def collect(env,w,robot,sensor,term,candidate,parent,teacher,gait):
 jobs=[]
 for split,count,base_seed in (("train",300,0),("validation",100,100000),("held_out",100,200000)):
  for d in range(0,360,45):
   for y in (-.3,0.,.3):
    for e in range(count):jobs.append((split,d,y,e,base_seed+(d//45)*10000+int((y+.3)*10)*1000+e))
 p=2;nmax=env.num_envs//p;cursor=0;buf={k:[] for k in ("observation","physical_command","actor_command","ramp_progress","previous_action","candidate_action","w1b_action","stop_action","base_velocity","base_angular_velocity","roll_pitch","foot_speed","contact_state")};meta=[];safe_obs=[];feet=sensor.find_bodies(".*_ankle_roll_link")[0];rfeet=robot.find_bodies(".*_ankle_roll_link")[0]
 while cursor<len(jobs):
  b=jobs[cursor:cursor+nmax];n=len(b);pad=b+[b[i%n] for i in range(nmax-n)] if n<nmax else b;ids=torch.arange(p*nmax,device=env.device);env.reset(env_ids=ids);term.external_override.zero_();term._update_command();obs=w.get_observations().to(env.device)
  for _ in range(round(3/env.step_dt)):
   with torch.inference_mode():a=teacher(obs["policy"],gait)
   obs,_,_,_=w.step(a);obs=obs.to(env.device)
  clone(env,robot,term,nmax,p);term.external_override.zero_();term._update_command();obs=w.get_observations().to(env.device)
  with torch.inference_mode():ca=candidate(obs["policy"],gait);ta=teacher(obs["policy"],gait)
  a=ca.clone();a[nmax:]=ta[nmax:];obs,_,_,_=w.step(a);obs=obs.to(env.device);dirs=torch.tensor([math.radians(x[1]) for x in pad],device=env.device);yb=torch.tensor([x[2] for x in pad],device=env.device);target=torch.stack((.3*dirs.cos(),.3*dirs.sin(),yb),1).repeat(p,1)
  batch={k:[] for k in buf};batch_safe=[]
  for k in range(1,5):
   alpha=minjerk(torch.tensor(k*env.step_dt/1.5,device=env.device));physical=target*alpha;term.external_override[:,:2]=physical[:,:2];term.external_override[:,2]=calibrate_yaw(physical[:,2]);term._update_command();obs=w.get_observations().to(env.device)
   with torch.inference_mode():ca=candidate(obs["policy"],gait);pa=parent(obs["policy"],gait);ta=teacher(obs["policy"],gait)
   sl=slice(0,n);force=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);contact=(force>5).to(torch.uint8);fs=torch.linalg.vector_norm(robot.data.body_lin_vel_w[:,rfeet,:2],dim=-1)
   batch["observation"].append(obs["policy"][sl].cpu());batch["physical_command"].append(physical[sl].cpu());batch["actor_command"].append(torch.cat((physical[sl,:2],calibrate_yaw(physical[sl,2])[:,None]),1).cpu());batch["ramp_progress"].append(torch.full((n,),k*env.step_dt/1.5));batch["previous_action"].append(env.action_manager.prev_action[sl].cpu());batch["candidate_action"].append(ca[sl].cpu());batch["w1b_action"].append(pa[sl].cpu());batch["stop_action"].append(ta[sl].cpu());batch["base_velocity"].append(robot.data.root_lin_vel_b[sl].cpu());batch["base_angular_velocity"].append(robot.data.root_ang_vel_b[sl].cpu());batch["roll_pitch"].append(robot.data.projected_gravity_b[sl,:2].cpu());batch["foot_speed"].append(fs[sl].cpu());batch["contact_state"].append(contact[sl].cpu());batch_safe.append(obs["policy"][nmax:nmax+n].cpu());a=ca.clone();a[nmax:]=pa[nmax:];obs,_,_,_=w.step(a);obs=obs.to(env.device)
  for key in buf:buf[key].append(torch.stack(batch[key],0))
  safe_obs.append(torch.stack(batch_safe,0));meta.extend(pad[:n]);cursor+=n;print(json.dumps({"collection":cursor,"total":len(jobs)}),flush=True)
 data={k:torch.cat(v,1) for k,v in buf.items()};data["safe_pc2_observation"]=torch.cat(safe_obs,1);split_code=torch.tensor([{"train":0,"validation":1,"held_out":2}[x[0]] for x in meta],dtype=torch.uint8);data.update({"split_code":split_code,"direction":torch.tensor([x[1] for x in meta]),"yaw":torch.tensor([x[2] for x in meta]),"episode_index":torch.tensor([x[3] for x in meta]),"seed":torch.tensor([x[4] for x in meta]),"condition_id":torch.tensor([(x[1]//45)*3+(-1 if x[2]<0 else 0 if x[2]==0 else 1)+1 for x in meta]),"boundary_steps":torch.tensor([1,2,3,4]),"runtime_action_source":"A4 V2 candidate only","label_source":"W1B-R2 deterministic mean"});torch.save(data,OUT/"start_boundary_trajectory_overlay_v3.pt")
 h=hashlib.sha256((OUT/"start_boundary_trajectory_overlay_v3.pt").read_bytes()).hexdigest();sem=hashlib.sha256()
 for k in sorted(x for x,v in data.items() if isinstance(v,torch.Tensor)):sem.update(k.encode());sem.update(data[k].contiguous().numpy().tobytes())
 counts={s:sum(x[0]==s for x in meta) for s in ("train","validation","held_out")};(OUT/"candidate_visited_boundary_collection_manifest.json").write_text(json.dumps({"conditions":24,"episodes":len(meta),"split_episode_counts":counts,"steps":["B0 state observed but not overlaid","B1","B2","B3","B4"],"runtime_action":"candidate only","W1B_usage":"label query only","seed_sets_disjoint":True},indent=2)+"\n");(OUT/"start_boundary_trajectory_overlay_v3_manifest.json").write_text(json.dumps({"name":"StartBoundaryTrajectoryOverlayV3","base":"immutable W2-P1 + StartBoundaryLabelContractV2","entries":len(meta)*4,"episode_count":len(meta),"byte_sha256":h,"semantic_sha256":sem.hexdigest(),"base_changed":0},indent=2)+"\n");(OUT/"start_boundary_trajectory_overlay_v3_split.json").write_text(json.dumps({"split_codes":{"train":0,"validation":1,"held_out":2},"episodes":counts,"entries":{k:v*4 for k,v in counts.items()},"seed_overlap":0},indent=2)+"\n")
def main():
 OUT.mkdir(parents=True,exist_ok=True);cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=args.max_envs;cfg.episode_length_s=12.;cfg.seed=20278301;cfg.observations.policy.enable_corruption=False
 if args.device:cfg.sim.device=agent.device=args.device
 with launch_simulation(cfg,args):
  w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);env=w.unwrapped;dev=env.device;candidate,rep,_,_,_=c.reproduce_a4(dev);parent=FrozenGaitActor(PARENT).to(dev).eval();teacher=FrozenGaitActor(TEACHER).to(dev).eval();robot=env.scene["robot"];sensor=env.scene.sensors["contact_forces"];term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True;gait=torch.zeros(env.num_envs,device=dev);gate=positive(env,w,robot,sensor,term,candidate,parent,teacher,gait)
  if gate:collect(env,w,robot,sensor,term,candidate,parent,teacher,gait)
  w.close();print(json.dumps({"PC2_pass":gate,"overlay_created":gate}),flush=True)
if __name__=="__main__":main()
