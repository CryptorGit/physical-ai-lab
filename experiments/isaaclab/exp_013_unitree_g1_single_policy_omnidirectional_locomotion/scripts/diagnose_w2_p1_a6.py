"""Teacher-forced rear-direction yaw acquisition diagnostics (read-only actors)."""
from __future__ import annotations
import argparse,csv,json,math,sys
from collections import defaultdict
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
import w2_p1_a5_common as common
BASE=common.BASE;OUT=BASE/"phase_w2_p1_a6_rear_yaw_acquisition_diagnosis";PARENT=BASE/"phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt";TEACHER=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
parser=argparse.ArgumentParser();parser.add_argument("--max-envs",type=int,default=4000);parser.add_argument("--section",choices=("exact","horizon","controls"),required=True);add_launcher_args(parser);args,hydra=setup_preset_cli(parser);sys.argv=[sys.argv[0],*hydra]
def minjerk(x):x=x.clamp(0,1);return x**3*(10-15*x+6*x*x)
def write_csv(name,rows):
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with (OUT/name).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def clone(env,robot,term,n,p):
 origins=env.scene.env_origins;refs=torch.arange(n,device=env.device);targets=torch.arange(p*n,device=env.device);src=targets.remainder(n);local=robot.data.root_pos_w[refs]-origins[refs];pose=torch.cat((local[src]+origins[targets],robot.data.root_quat_w[refs][src]),1);vel=torch.cat((robot.data.root_lin_vel_w[refs][src],robot.data.root_ang_vel_w[refs][src]),1);robot.write_root_pose_to_sim(pose,targets);robot.write_root_velocity_to_sim(vel,targets);robot.write_joint_state_to_sim(robot.data.joint_pos[refs][src],robot.data.joint_vel[refs][src],env_ids=targets)
 for _,v in vars(term).items():
  if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==env.num_envs:v[targets]=v[refs][src].clone()
 env.action_manager._action[targets]=env.action_manager.action[refs][src];env.action_manager._prev_action[targets]=env.action_manager.prev_action[refs][src];env.episode_length_buf[targets]=env.episode_length_buf[refs][src];env.sim.forward()
def profile_action(defn,k,ca,pa,ta):
 if k==0:return pa if defn.get("b0")=="parent" else ca if defn.get("b0")=="candidate" else ta
 kind=defn["kind"]
 if kind=="candidate":return ca
 if kind=="steps":return pa if k<=defn["steps"] else ca
 if kind in ("full_ramp","full_start","w1b_only"):return pa
 return ca
def run_matrix(env,w,robot,sensor,term,candidate,parent,teacher,gait,jobs,profiles,total_after_ramp=8.0,tag="run"):
 p=len(profiles);nmax=env.num_envs//p;rows=[];snap=[];bins=defaultdict(lambda:{"n":0,"dir":0.,"vec":0.,"wrong":0,"fail":0});cursor=0;feet=sensor.find_bodies(".*_ankle_roll_link")[0];rfeet=robot.find_bodies(".*_ankle_roll_link")[0];dt=env.step_dt
 while cursor<len(jobs):
  b=jobs[cursor:cursor+nmax];n=len(b);pad=b+[b[i%n] for i in range(nmax-n)] if n<nmax else b;ids=torch.arange(p*nmax,device=env.device);env.reset(env_ids=ids);term.external_override.zero_();term._update_command();obs=w.get_observations().to(env.device)
  for _ in range(round(3/dt)):
   with torch.inference_mode():a=teacher(obs["policy"],gait)
   obs,_,_,_=w.step(a);obs=obs.to(env.device)
  clone(env,robot,term,nmax,p);term.external_override.zero_();term._update_command();obs=w.get_observations().to(env.device)
  dirs=torch.tensor([math.radians(x["direction"]) for x in pad],device=env.device);speed=torch.tensor([x["speed"] for x in pad],device=env.device);yb=torch.tensor([x["yaw"] for x in pad],device=env.device);base_target=torch.stack((speed*dirs.cos(),speed*dirs.sin(),yb),1).repeat(p,1)
  ramp=torch.cat([torch.full((nmax,),d.get("ramp",1.5),device=env.device) for d in profiles]);ramp_end=(ramp/dt).round().long();maxsteps=round((max(d.get("ramp",1.5) for d in profiles)+total_after_ramp)/dt)
  names=("translation","direction","yaw","gait","combined_a5","combined_formal");streak={x:torch.zeros(env.num_envs,dtype=torch.long,device=env.device) for x in names};maxst={x:torch.zeros_like(streak[x]) for x in names};resets={x:torch.zeros_like(streak[x]) for x in names};first={x:torch.full((env.num_envs,),float("nan"),device=env.device) for x in names};first10={x:torch.full_like(first[x],float("nan")) for x in names};first20={x:torch.full_like(first[x],float("nan")) for x in names};first30={x:torch.full_like(first[x],float("nan")) for x in names};first50={x:torch.full_like(first[x],float("nan")) for x in names};fall=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device);slip=torch.zeros_like(fall);impact=torch.zeros_like(fall);slipst=torch.zeros_like(streak["yaw"]);end_sum=torch.zeros(env.num_envs,device=env.device);end_n=torch.zeros_like(end_sum);contact_switch=torch.zeros_like(streak["yaw"]);prev_support=torch.full_like(streak["yaw"],-1)
  for k in range(0,maxsteps+1):
   alpha=minjerk(torch.tensor(k*dt,device=env.device)/ramp);physical=base_target*alpha[:,None];term.external_override[:,:2]=physical[:,:2];term.external_override[:,2]=calibrate_yaw(physical[:,2]);term._update_command();obs=w.get_observations().to(env.device)
   with torch.inference_mode():ca=candidate(obs["policy"],gait);pa=parent(obs["policy"],gait);ta=teacher(obs["policy"],gait)
   acts=[]
   for bi,d in enumerate(profiles):acts.append(profile_action(d,k,ca[bi*nmax:(bi+1)*nmax],pa[bi*nmax:(bi+1)*nmax],ta[bi*nmax:(bi+1)*nmax]))
   action=torch.cat(acts);prev=env.action_manager.prev_action.clone();obs,_,dn,x=w.step(action);obs=obs.to(env.device);fall|=dn.bool()&~x.get("time_outs",torch.zeros_like(dn)).bool();force=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);contact=force>5;fs=torch.linalg.vector_norm(robot.data.body_lin_vel_w[:,rfeet,:2],dim=-1);bad=(fs>.55)&contact;slipst=torch.where(bad.any(1),slipst+1,torch.zeros_like(slipst));slip|=slipst>=5;impact|=force.amax(1)>3500;support=torch.where(contact[:,0]&~contact[:,1],0,torch.where(contact[:,1]&~contact[:,0],1,torch.where(contact.all(1),2,3)));contact_switch+=(support!=prev_support)&(prev_support>=0);prev_support=support
   actual=robot.data.root_lin_vel_b[:,:2];ay=robot.data.root_ang_vel_b[:,2];final=base_target;vec=torch.linalg.vector_norm(actual-final[:,:2],dim=1);aspeed=torch.linalg.vector_norm(actual,dim=1);taang=torch.atan2(final[:,1],final[:,0]);aa=torch.atan2(actual[:,1],actual[:,0]);de=(torch.atan2(torch.sin(aa-taang),torch.cos(aa-taang)).abs()*180/math.pi);trans=torch.where(torch.linalg.vector_norm(final[:,:2],dim=1)<=1e-8,aspeed<=.08,vec<=.25);dirok=torch.where(torch.linalg.vector_norm(final[:,:2],dim=1)<=1e-8,torch.ones_like(trans),de<=25);yawerr=(ay-final[:,2]).abs();yawok=torch.where(final[:,2].abs()<=1e-8,ay.abs()<=.2,(torch.sign(ay)==torch.sign(final[:,2]))&(yawerr<=.2));gaitok=contact.any(1);safe=~fall&~slip&~impact;mask={"translation":trans,"direction":dirok,"yaw":yawok,"gait":gaitok,"combined_a5":trans&yawok&safe,"combined_formal":trans&dirok&yawok&gaitok&safe}
   after=k>=ramp_end
   rel=k*dt-ramp
   for nm,m in mask.items():
    active=m&after;first[nm][torch.isnan(first[nm])&active]=rel[torch.isnan(first[nm])&active];resets[nm]+=((~active)&(streak[nm]>0)&after);streak[nm]=torch.where(active,streak[nm]+1,torch.zeros_like(streak[nm]));maxst[nm]=torch.maximum(maxst[nm],streak[nm])
    for dur,dest in ((.1,first10[nm]),(.2,first20[nm]),(.3,first30[nm]),(.5,first50[nm])):
     req=round(dur/dt);hit=torch.isnan(dest)&(streak[nm]>=req);dest[hit]=rel[hit]-(req-1)*dt
   endpoint_window=(rel>=2.0)&(rel<=4.0);end_sum+=mask["combined_a5"].float()*endpoint_window;end_n+=endpoint_window
   # Detailed low-speed bins for PC2 focus conditions.
   if tag=="pc_exact":
    for bi,d in enumerate(profiles):
     if d["name"]!="PC2":continue
     sl=slice(bi*nmax,bi*nmax+n)
     for lo,hi,label in ((0,.025,"0-.025"),(.025,.05,".025-.05"),(.05,.10,".05-.10"),(.10,.15,".10-.15"),(.15,.20,".15-.20"),(.20,.30,".20-.30"),(.30,99,">.30")):
      sel=(aspeed[sl]>=lo)&(aspeed[sl]<hi)&after[sl]
      if sel.any():q=bins[label];q["n"]+=int(sel.sum());q["dir"]+=float(de[sl][sel].sum());q["vec"]+=float(vec[sl][sel].sum());q["wrong"]+=int((~dirok[sl][sel]).sum());q["fail"]+=int((~mask["combined_formal"][sl][sel]).sum())
   # Takeover snapshots.
   for bi,d in enumerate(profiles):
    take=d.get("takeover",d.get("steps",-99)+1 if d["kind"]=="steps" else -99)
    if k in (take-1,take,take+1,take+2,take+4,take+8,take+16):
     sl=slice(bi*nmax,bi*nmax+n);snap.append({"run":tag,"profile":d["name"],"batch_start":cursor,"relative_step":k-take,"samples":n,"action_l2_W1B_candidate":float(torch.linalg.vector_norm(pa[sl]-ca[sl],dim=1).mean()),"action_cosine":float(torch.nn.functional.cosine_similarity(pa[sl],ca[sl]).mean()),"speed":float(aspeed[sl].mean()),"direction_error":float(de[sl].mean()),"yaw_error":float(yawerr[sl].mean()),"contact_switches_mean":float(contact_switch[sl].float().mean())})
  for bi,d in enumerate(profiles):
   for i,jb in enumerate(pad[:n]):
    j=bi*nmax+i;r={"run":tag,"profile":d["name"],"direction":jb["direction"],"speed_target":jb["speed"],"yaw":jb["yaw"],"episode":jb["episode"],"endpoint_success":int(end_n[j]>0 and end_sum[j]/end_n[j]>=.999),"fall":int(fall[j]),"dangerous_slip":int(slip[j]),"impact":int(impact[j]),"contact_switches":int(contact_switch[j])}
    for nm in names:
     r[f"first_{nm}_s"]=None if torch.isnan(first[nm][j]) else float(first[nm][j]);r[f"first_0p10_{nm}_s"]=None if torch.isnan(first10[nm][j]) else float(first10[nm][j]);r[f"first_0p20_{nm}_s"]=None if torch.isnan(first20[nm][j]) else float(first20[nm][j]);r[f"first_0p30_{nm}_s"]=None if torch.isnan(first30[nm][j]) else float(first30[nm][j]);r[f"first_0p50_{nm}_s"]=None if torch.isnan(first50[nm][j]) else float(first50[nm][j]);r[f"resets_{nm}"]=int(resets[nm][j]);r[f"longest_{nm}_s"]=float(maxst[nm][j]*dt)
    rows.append(r)
  cursor+=n;print(json.dumps({tag:cursor,"total":len(jobs)}),flush=True)
 return rows,snap,bins
def main():
 OUT.mkdir(parents=True,exist_ok=True);cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=min(args.max_envs,4000 if args.section=="exact" else 3960);cfg.episode_length_s=12. if args.section=="exact" else 16.;cfg.seed=20278301 if args.section=="exact" else 20278401;cfg.observations.policy.enable_corruption=False
 if args.device:cfg.sim.device=agent.device=args.device
 allrows=[];allsnap=[];allbins={}
 with launch_simulation(cfg,args):
  w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);env=w.unwrapped;dev=env.device;candidate,rep,_,_,_=common.reproduce_a4(dev);parent=FrozenGaitActor(PARENT).to(dev).eval();teacher=FrozenGaitActor(TEACHER).to(dev).eval();robot=env.scene["robot"];sensor=env.scene.sensors["contact_forces"];term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True;gait=torch.zeros(env.num_envs,device=dev)
  if args.section=="exact":
   jobs=[{"direction":d,"speed":.3,"yaw":y,"episode":e} for d in range(0,360,45) for y in (-.3,0.,.3) for e in range(200)];profiles=[{"name":"PC0","kind":"candidate","b0":"candidate"},{"name":"PC1","kind":"steps","steps":2,"b0":"teacher"},{"name":"PC2","kind":"steps","steps":4,"b0":"teacher"},{"name":"PC3","kind":"steps","steps":4,"b0":"parent"}];r,s,b=run_matrix(env,w,robot,sensor,term,candidate,parent,teacher,gait,jobs,profiles,4.,"pc_exact");allrows+=r;allsnap+=s;allbins=b
  elif args.section=="horizon":
   focus=[(180,.3),(180,-.3),(135,.3),(225,-.3),(0,.3),(0,-.3),(90,.3),(270,-.3)];jobs=[{"direction":d,"speed":.3,"yaw":y,"episode":e} for d,y in focus for e in range(200)];profiles=[{"name":f"W1B_{h}_STEPS","kind":"steps","steps":h,"b0":"teacher"} for h in (2,4,6,8,12,16)]+[{"name":"FULL_RAMP","kind":"steps","steps":75,"b0":"teacher"},{"name":"FULL_START","kind":"steps","steps":275,"b0":"teacher"},{"name":"W1B_ONLY","kind":"w1b_only","b0":"teacher"}];r,s,_=run_matrix(env,w,robot,sensor,term,candidate,parent,teacher,gait,jobs,profiles,8.,"horizon");allrows+=r;allsnap+=s
  else:
   focus=[(180,.3),(180,-.3),(135,.3),(225,-.3)];jobs=[{"direction":d,"speed":.3,"yaw":y,"episode":e} for d,y in focus for e in range(200)];profiles=[{"name":f"RAMP_{x}","kind":"steps","steps":4,"b0":"teacher","ramp":x} for x in (.75,1.,1.5,2.,3.)];r,s,_=run_matrix(env,w,robot,sensor,term,candidate,parent,teacher,gait,jobs,profiles,6.,"ramp");allrows+=r;allsnap+=s
   fac=[(180,.3,0),(180,.3,.3),(180,.3,-.3),(180,.15,.3),(180,.15,-.3),(0,0,.3),(0,0,-.3),(0,.3,.3),(0,.3,-.3)];jobs=[{"direction":d,"speed":sp,"yaw":y,"episode":e} for d,sp,y in fac for e in range(200)];profiles=[{"name":"PC2_FACTORIAL","kind":"steps","steps":4,"b0":"teacher"}];r,s,_=run_matrix(env,w,robot,sensor,term,candidate,parent,teacher,gait,jobs,profiles,8.,"factorial");allrows+=r;allsnap+=s
  write_csv(f"raw_a6_{args.section}_episode_metrics.csv",allrows);write_csv(f"raw_a6_{args.section}_takeover.csv",allsnap);(OUT/f"raw_a6_{args.section}_low_speed_bins.json").write_text(json.dumps(allbins,indent=2)+"\n");print(json.dumps({"section":args.section,"rows":len(allrows),"snapshots":len(allsnap),"candidate_hash":rep["tensor_hash"]}),flush=True);w.close()
if __name__=="__main__":main()
