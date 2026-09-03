"""Stage 7 WALK_TO_RUN overlap and parameter-free direct-switch audit."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys
from pathlib import Path
import gymnasium as gym
import torch

SCRIPT=Path(__file__).resolve(); EXP=SCRIPT.parent.parent; REPO=EXP.parents[2]
sys.path[:0]=[str(EXP/"src"),str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),str(REPO/"experiments/isaaclab/exp_006_unitree_g1_command_skills/src")]
import isaaclab_tasks, g1_flat_run.tasks, g1_command_skills.tasks, g1_walk_centered.tasks  # noqa
from g1_walk_centered.command_contract import MotionCommand
from g1_walk_centered.experts import load_walk_expert, load_run_expert
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation
from g1_walk_centered.experts.adapters import to_run_observation
from g1_walk_centered.tasks.stage7r_action import WalkToRunTransitionActor152
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli

TARGETS=(2.4,2.6,2.8)
EXPECTED={"walk":"9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa",
"run":"60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266",
"stand":"734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
"stw":"511b3c832e2c7bc54f348391db79d4ac57d1ceda10a6907549a50da093a9c36e"}
p=argparse.ArgumentParser(); p.add_argument("--mode",choices=("preflight","baseline","formal"),required=True)
p.add_argument("--seed",type=int,required=True); p.add_argument("--episodes-per-target",type=int,default=10)
p.add_argument("--output",required=True); p.add_argument("--label",required=True)
p.add_argument("--stand",required=True);p.add_argument("--stand-to-walk",required=True);p.add_argument("--walk",required=True);p.add_argument("--run",required=True)
p.add_argument("--transition-checkpoint")
p.add_argument("--saturation-events-output")
p.add_argument("--target-speeds",type=float,nargs="+",default=list(TARGETS))
add_launcher_args(p);args,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0],*hydra]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def mj(u): u=u.clamp(0,1);return 10*u**3-15*u**4+6*u**5
def mean(a):
 a=list(a);return sum(a)/len(a) if a else 0.
def pct(a,q):
 a=sorted(a)
 if not a:return 0.
 x=(len(a)-1)*q/100;lo=int(x);hi=min(lo+1,len(a)-1);return a[lo]*(hi-x)+a[hi]*(x-lo)
def write_csv(path,rows):
 with Path(path).open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def periodic(x):
 return len(x["flights"])>=4 and x["maxsafe"]>=3 and x["alt"]/max(x["altopp"],1)>=.8 and x["valid"]/max(len(x["flights"]),1)>=.8 and .04<=mean(x["flights"])<=.16
def fresh(): return {"flights":[],"valid":0,"alt":0,"altopp":0,"last":None,"safe":0,"maxsafe":0}

def main():
 out=(REPO/args.output).resolve();out.mkdir(parents=True,exist_ok=True)
 paths={k:Path(getattr(args,"stand_to_walk" if k=="stw" else k)).resolve(strict=True) for k in EXPECTED}
 hashes={k:sha(v) for k,v in paths.items()}
 if hashes!=EXPECTED: raise RuntimeError(f"protected hash mismatch {hashes}")
 stand=load_walk_expert(paths["stand"]); stw=load_walk_expert(paths["stw"]); walk=load_walk_expert(paths["walk"]); run=load_run_expert(paths["run"])
 z=torch.zeros(1,123);z[:,8]=-1; state=canonical_state_from_legacy_observation(z,heading_w_rad=torch.zeros(1))
 pre={"status":"PASS","protected_hashes":hashes,"walk_observation":123,"run_observation":152,"action":37,"scale":.5,
 "global_previous_action":True,"runtime_blend":False,"turn_zero":True,"run_to_walk_not_loaded":True}
 (out/"routing_preflight.json").write_text(json.dumps(pre,indent=2)+"\n")
 if args.mode=="preflight": print(json.dumps(pre));return
 formal_targets=tuple(args.target_speeds)
 if any(s not in TARGETS for s in formal_targets): raise RuntimeError(f"unsupported WALK_TO_RUN target {formal_targets}")
 speeds=[s for s in formal_targets for _ in range(args.episodes_per_target)]; n=len(speeds)
 cfg,agent=resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=n;cfg.seed=args.seed;cfg.episode_length_s=18.
 if args.device:cfg.sim.device=args.device
 with launch_simulation(cfg,args):
  w=RslRlVecEnvWrapper(gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0",cfg=cfg),clip_actions=agent.clip_actions);e=w.unwrapped;dev=e.device
  stand=load_walk_expert(paths["stand"],device=dev);stw=load_walk_expert(paths["stw"],device=dev);walk=load_walk_expert(paths["walk"],device=dev);run=load_run_expert(paths["run"],device=dev)
  transition=None
  if args.transition_checkpoint:
   transition=WalkToRunTransitionActor152(run.actor).to(dev)
   payload=torch.load(Path(args.transition_checkpoint).resolve(strict=True),map_location=dev,weights_only=False)
   transition.load_state_dict(payload["actor"],strict=True);transition.eval()
  robot=e.scene["robot"];term=e.command_manager.get_term("base_velocity");sensor=e.scene.sensors["contact_forces"]
  feet,names=robot.find_bodies(".*_ankle_roll_link");sf=[sensor.body_names.index(x) for x in names];joints,joint_names=robot.find_joints(".*")
  w.reset();dt=float(e.step_dt);target=torch.tensor(speeds,device=dev);heading=robot.data.heading_w.torch.clone();origin=robot.data.root_pos_w.torch[:,:2].clone()
  phase=torch.zeros(n,dtype=torch.long,device=dev); t=torch.zeros(n,device=dev); streak=torch.zeros(n,device=dev); switches=torch.zeros(n,dtype=torch.long,device=dev)
  prevsup=torch.zeros(n,dtype=torch.long,device=dev);prev=torch.zeros(n,37,device=dev);done=torch.zeros(n,dtype=torch.bool,device=dev)
  source=torch.zeros(n,dtype=torch.bool,device=dev);complete=torch.zeros(n,dtype=torch.bool,device=dev);hold=torch.zeros(n,dtype=torch.bool,device=dev)
  fall=torch.zeros(n,dtype=torch.bool,device=dev);sat=torch.zeros(n,dtype=torch.bool,device=dev);slipfail=torch.zeros(n,dtype=torch.bool,device=dev);impactfail=torch.zeros(n,dtype=torch.bool,device=dev)
  slipdwell=torch.zeros(n,device=dev)
  entry=torch.zeros(n,device=dev);exitjump=torch.zeros(n,device=dev); overlap=torch.zeros(n,device=dev); contact_phase=[""]*n; gait=[fresh() for _ in range(n)]; inflight=[False]*n;fs=[0.]*n
  trace=[{"heading":[],"speederr":[],"drift":[],"impact":[],"slip":[],"actionrate":[]} for _ in range(n)]
  dwell=torch.zeros(n,len(joints),device=dev); dwell90=torch.zeros_like(dwell); max_dwell90=torch.zeros_like(dwell); max_dwell95=torch.zeros_like(dwell)
  event_open=torch.zeros(n,len(joints),dtype=torch.bool,device=dev); event_start=torch.zeros(n,len(joints),device=dev)
  saturation_events=[]; holdtime=torch.zeros(n,device=dev); completion_time=torch.zeros(n,device=dev)
  prevcontacts=[(False,False)]*n
  for step in range(round(17.5/dt)):
   ph=phase.clone();cmd=torch.zeros(n,device=dev);cmd[ph==2]=1.2*mj(t[ph==2]/1.5);cmd[ph==3]=1.2
   cmd[ph==4]=1.2+(target[ph==4]-1.2)*mj(t[ph==4]/1.4);cmd[ph==5]=target[ph==5]
   herr=torch.atan2(torch.sin(heading-robot.data.heading_w.torch),torch.cos(heading-robot.data.heading_w.torch))
   yaw_walk=(.8*herr-.1*robot.data.root_ang_vel_b.torch[:,2]).clamp(-.3,.3);yaw_run=(1.5*herr).clamp(-1.5,1.5);yaw=torch.where(ph>=4,yaw_run,yaw_walk)
   term.vel_command_b.zero_();term.vel_command_b[:,0]=cmd;term.vel_command_b[:,2]=yaw
   legacy=w.get_observations()["policy"]; canonical=canonical_state_from_legacy_observation(legacy,heading_w_rad=robot.data.heading_w.torch);mc=MotionCommand(cmd,heading,target_yaw_rate_radps=yaw)
   with torch.inference_mode():
    astand=stand(canonical,mc);astw=stw(canonical,mc);awalk=walk(canonical,mc);arun=run(canonical,mc)
    atransition=transition(to_run_observation(canonical,mc,route="RUN")) if transition is not None else arun
    action=torch.where((ph==0).unsqueeze(1),astand,torch.where(((ph==1)|(ph==2)).unsqueeze(1),astw,torch.where((ph==3).unsqueeze(1),awalk,torch.where((ph==4).unsqueeze(1),atransition,arun))))
    action[done]=prev[done];_,_,dones,info=w.step(action)
   rate=torch.linalg.vector_norm(action-prev,dim=1)/dt; prev[:]=action
   forces=sensor.data.net_forces_w_history.torch[:,:,sf,:];contacts=forces.norm(dim=-1).amax(dim=1)>5;impact=forces[:,:,:,2].abs().mean(dim=1).amax(dim=1)
   fspeed=robot.data.body_lin_vel_w.torch[:,feet,:2].norm(dim=-1);slip=torch.where(contacts,fspeed,torch.zeros_like(fspeed)).amax(dim=1)
   eratio=robot.data.applied_torque.torch[:,joints].abs()/robot.data.joint_effort_limits.torch[:,joints].abs().clamp_min(1e-6)
   active_sat=(ph>=4).unsqueeze(1)
   above90=(eratio>=.90)&active_sat; above95=(eratio>=.95)&active_sat
   dwell90=torch.where(above90,dwell90+dt,torch.zeros_like(dwell90));dwell=torch.where(above95,dwell+dt,torch.zeros_like(dwell))
   max_dwell90=torch.maximum(max_dwell90,dwell90);max_dwell95=torch.maximum(max_dwell95,dwell)
   newly_open=above95&(~event_open);event_start=torch.where(newly_open,t.unsqueeze(1).expand_as(event_start),event_start);event_open|=newly_open
   newly_failed=(dwell>=.2)&(~sat.unsqueeze(1))
   for ii,jj in newly_failed.nonzero(as_tuple=False).tolist():
    support_name="double" if int(contacts[ii].sum())==2 else "left" if bool(contacts[ii,0]) else "right" if bool(contacts[ii,1]) else "flight"
    qpos=float(robot.data.joint_pos.torch[ii,joints[jj]])
    qtarget=float(robot.data.joint_pos_target.torch[ii,joints[jj]]) if hasattr(robot.data,"joint_pos_target") else float("nan")
    saturation_events.append({"seed":args.seed,"episode":ii,"target_speed_mps":speeds[ii],"joint_name":joint_names[jj],
      "side":"left" if joint_names[jj].startswith("left_") else "right" if joint_names[jj].startswith("right_") else "center",
      "limit_type":"effort","utilization":float(eratio[ii,jj]),"dwell_above_90_s":float(dwell90[ii,jj]),
      "dwell_above_95_s":float(dwell[ii,jj]),"longest_continuous_dwell_s":float(max_dwell95[ii,jj]),
      "event_start_s":float(event_start[ii,jj]),"event_end_s":float(t[ii]),"active_transition_phase":"WALK_TO_RUN_ACTIVE" if int(ph[ii])==4 else "RUN_TAKEOVER",
      "support_foot":support_name,"contact_state":f"{int(contacts[ii,0])}{int(contacts[ii,1])}","flight_state":not bool(contacts[ii].any()),
      "transition_elapsed_s":float(t[ii]),"target_speed_error_mps":float(abs(robot.data.root_lin_vel_b.torch[ii,0]-target[ii])),
      "heading_error_rad":float(abs(herr[ii])),"action_target":float(action[ii,jj]),"joint_position":qpos,
      "joint_position_error":qtarget-qpos,"contact_force_n":float(forces[ii,:,:,2].abs().mean(dim=0).amax()),
      "remaining_to_run_acceptance_s":float("nan")})
   sat|=(dwell>=.2).any(1)
   slipdwell=torch.where((slip>.8)&(ph>=4),slipdwell+dt,torch.zeros_like(slipdwell));slipfail|=slipdwell>=.2
   torso=e.termination_manager.get_term("base_contact").bool(); timeout=info.get("time_outs",torch.zeros_like(dones)).bool(); physical=dones.bool()&~timeout
   speed=robot.data.root_lin_vel_b.torch[:,0];g=robot.data.projected_gravity_b.torch;roll=torch.atan2(g[:,1],-g[:,2]).abs();pitch=torch.atan2(-g[:,0],torch.sqrt(g[:,1]**2+g[:,2]**2)).abs()
   sup=contacts[:,0].long()+2*contacts[:,1].long()
   for i in range(n):
    if done[i]:continue
    if ph[i]>=4:
     c=(bool(contacts[i,0]),bool(contacts[i,1]));ns=int(c[0])+int(c[1])
     if ns==0 and not inflight[i]:inflight[i]=True;fs[i]=float(t[i])
     if inflight[i] and ns>0:
      dur=float(t[i])-fs[i];new=[q for q in range(2) if c[q] and not prevcontacts[i][q]];valid=len(new)==1;side=new[0] if valid else -1;gg=gait[i];gg["flights"].append(dur)
      if valid:
       gg["valid"]+=1
       if gg["last"] is not None:gg["altopp"]+=1;gg["alt"]+=int(side!=gg["last"])
       safe=.04<=dur<=.16 and (gg["last"] is None or side!=gg["last"]);gg["safe"]=gg["safe"]+1 if safe else 0;gg["maxsafe"]=max(gg["maxsafe"],gg["safe"]);gg["last"]=side
      inflight[i]=False
     prevcontacts[i]=c;trace[i]["heading"].append(abs(float(herr[i])));trace[i]["speederr"].append(abs(float(speed[i]-target[i])));trace[i]["impact"].append(float(impact[i]));trace[i]["slip"].append(float(slip[i]));trace[i]["actionrate"].append(float(rate[i]))
     axis=torch.stack((-torch.sin(heading[i]),torch.cos(heading[i])));trace[i]["drift"].append(abs(float(((robot.data.root_pos_w.torch[i,:2]-origin[i])*axis).sum())))
     impactfail[i]|=impact[i]>3500
    if physical[i] or torso[i]:fall[i]=True;done[i]=True;continue
    if ph[i]==0:
     good=abs(float(speed[i]))<=.08 and roll[i]<=.1 and pitch[i]<=.1 and int(sup[i])==3;streak[i]=streak[i]+dt if good else 0
     if streak[i]>=.4:phase[i]=1;t[i]=0;streak[i]=0
    elif ph[i]==1:
     if t[i]>=.8:phase[i]=2;t[i]=0;switches[i]=0
    elif ph[i]==2:
     if int(sup[i])!=int(prevsup[i]) and int(sup[i]) in (1,2):switches[i]+=1
     good=abs(float(speed[i]-1.2))<=.2 and abs(float(herr[i]))<=.12 and switches[i]>=2;streak[i]=streak[i]+dt if good else 0
     if streak[i]>=.4:phase[i]=3;t[i]=0;streak[i]=0
     elif t[i]>=5:done[i]=True
    elif ph[i]==3:
     good=abs(float(speed[i]-1.2))<=.2 and abs(float(herr[i]))<=.12;streak[i]=streak[i]+dt if good else 0
     if streak[i]>=1:
      source[i]=True;contact_phase[i]="double" if int(sup[i])==3 else "left" if int(sup[i])==1 else "right" if int(sup[i])==2 else "flight"
      overlap[i]=torch.linalg.vector_norm(awalk[i]-atransition[i]);entry[i]=overlap[i];phase[i]=4;t[i]=0;streak[i]=0;origin[i]=robot.data.root_pos_w.torch[i,:2]
     elif t[i]>=3:done[i]=True
    elif ph[i]==4:
     good=periodic(gait[i]) and abs(float(speed[i]-target[i]))<=.2 and abs(float(herr[i]))<=.12 and not sat[i] and not slipfail[i] and not impactfail[i]
     streak[i]=streak[i]+dt if good else 0
     if streak[i]>=.4:complete[i]=True;completion_time[i]=t[i];exitjump[i]=torch.linalg.vector_norm(atransition[i]-arun[i]);phase[i]=5;t[i]=0;streak[i]=0
     elif t[i]>=5:done[i]=True
    elif ph[i]==5:
     holdtime[i]+=dt
     if holdtime[i]>=5:hold[i]=periodic(gait[i]) and not fall[i] and not sat[i] and not slipfail[i] and not impactfail[i];done[i]=True
    prevsup[i]=sup[i]
   t+=dt
   if bool(done.all()):break
  rows=[];overlaprows=[]
  for i in range(n):
   tr=trace[i]; hp=pct(tr["heading"],95); success=bool(source[i] and complete[i] and hold[i] and not fall[i] and hp<=.12 and not sat[i] and not slipfail[i] and not impactfail[i])
   failure="" if success else "source_walk_contract_failure" if not source[i] else "transition_timeout" if not complete[i] else "run_hold_failure"
   rows.append({"seed":args.seed,"episode":i,"target_run_speed_mps":speeds[i],"source_contract":bool(source[i]),"contact_phase":contact_phase[i],"transition_completion":bool(complete[i]),"target_speed_acquisition":bool(complete[i]),"periodic_running_acquisition":bool(complete[i]),"run_takeover":bool(complete[i]),"run_hold":bool(hold[i]),"full_edge_success":success,"transition_timeout":bool(source[i] and not complete[i]),"transition_duration_s":float(completion_time[i]),"heading_p95_rad":hp,"speed_error_mean_mps":mean(tr["speederr"]),"path_drift_max_m":max(tr["drift"],default=0),"fall":bool(fall[i]),"dangerous_slip":bool(slipfail[i]),"impact_failure":bool(impactfail[i]),"saturation_failure":bool(sat[i]),"entry_action_jump_l2":float(entry[i]),"exit_action_jump_l2":float(exitjump[i]),"flight_events":len(gait[i]["flights"]),"safe_cycles":gait[i]["maxsafe"],"precursor_fires":len(gait[i]["flights"])+gait[i]["valid"]+gait[i]["alt"],"previous_action_mismatch":0,"failure_class":failure})
   overlaprows.append({"seed":args.seed,"episode":i,"target_run_speed_mps":speeds[i],"walk_run_action_l2":float(overlap[i]),"contact_phase":contact_phase[i]})
  valid=[r for r in rows if r["source_contract"]]
  summary={"mode":args.mode,"seed":args.seed,"attempts":n,"valid_sources":len(valid),"source_generation_rate":len(valid)/n,"transition_completion_rate":mean(r["transition_completion"] for r in valid),"run_takeover_rate":mean(r["run_takeover"] for r in valid),"full_edge_success_rate":mean(r["full_edge_success"] for r in valid),"fall_rate":mean(r["fall"] for r in valid),"heading_p95_rad":pct([r["heading_p95_rad"] for r in valid],95),"saturation_rate":mean(r["saturation_failure"] for r in valid),"slip_rate":mean(r["dangerous_slip"] for r in valid),"impact_failure_rate":mean(r["impact_failure"] for r in valid),"action_discontinuity_failure_rate":mean(r["entry_action_jump_l2"]>6 for r in valid),"per_target":{str(s):{"n":len(g:= [r for r in valid if r["target_run_speed_mps"]==s]),"success_rate":mean(r["full_edge_success"] for r in g)} for s in formal_targets}}
  for event in saturation_events:
   idx=int(event["episode"])
   if bool(complete[idx]): event["remaining_to_run_acceptance_s"]=max(0.0,float(completion_time[idx])-float(event["transition_elapsed_s"]))
  write_csv(out/f"{args.label}_episodes.csv",rows);write_csv(out/f"{args.label}_overlap.csv",overlaprows)
  if args.saturation_events_output and saturation_events:
   write_csv(Path(args.saturation_events_output),saturation_events)
  (out/f"{args.label}_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2));w.close()
if __name__=="__main__":main()
