"""D6 read-only omnidirectional WALK-to-STAND route audit."""
from __future__ import annotations
import argparse,copy,csv,hashlib,importlib.util,json,math,sys
from pathlib import Path
import gymnasium as gym
import torch

HERE=Path(__file__).resolve();EXP=HERE.parent.parent;REPO=EXP.parents[2];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d6_omnidirectional_stop_teacher";RAW=OUT/"raw"
spec=importlib.util.spec_from_file_location("d3",HERE.parent/"run_phase2_d3.py");d3=importlib.util.module_from_spec(spec);spec.loader.exec_module(d3)
from g1_explicit_motion_mode.contract import MotionMode,minimum_jerk  # noqa:E402
from g1_omnidirectional.policy import FrozenGaitActor  # noqa:E402
from g1_omnidirectional.yaw_calibration import calibrate_yaw  # noqa:E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa:E402
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli  # noqa:E402

DT=.02;ROUTES=("R0_W_MOVE_ZERO","R1_STAGE2Q_STOP","R2_S_HOLD_DIRECT","R3_W_MOVE_THEN_S_HOLD","R4_W_MOVE_STAGE2Q_HOLD")
def conditions():
 x=[{"condition_id":i,"kind":"zero_yaw","direction_deg":22.5*i,"speed":.3,"yaw":0.} for i in range(16)]
 for d in range(8):
  for y in (-.3,.3):x.append({"condition_id":len(x),"kind":"moving_yaw","direction_deg":45.*d,"speed":.3,"yaw":y})
 for y in (-.3,.3):x.append({"condition_id":len(x),"kind":"pure_yaw","direction_deg":0.,"speed":0.,"yaw":y})
 return x
def sha_bytes(*xs):
 h=hashlib.sha256()
 for x in xs:h.update(x.detach().contiguous().cpu().numpy().tobytes())
 return h.hexdigest()
def dump(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def rate(x):return float(torch.tensor(x,dtype=torch.float32).mean()) if x else 0.

def command_matrix(entries,count,dev):
 target=torch.zeros(count,3,device=dev)
 for j,e in enumerate(entries):r=math.radians(e["condition"]["direction_deg"]);target[j,0]=e["condition"]["speed"]*math.cos(r);target[j,1]=e["condition"]["speed"]*math.sin(r);target[j,2]=e["condition"]["yaw"]
 return target
def set_command(world,physical):
 actor=physical.clone();actor[:,2]=calibrate_yaw(actor[:,2]);world.term.external_override[:,:3]=actor;world.term._update_command()

def generate_batch(world,walk,entries,batch_id,path=None):
 n=len(entries);recipes=[e["recipe_id"] for e in entries];pad=recipes+[recipes[-1]]*(world.env.num_envs-n);world.restore(torch.tensor(pad,device=world.device));dev=world.device;world.state.request(torch.full((world.env.num_envs,),int(MotionMode.WALK),device=dev));target=command_matrix(entries,world.env.num_envs,dev);gait=torch.zeros(world.env.num_envs,device=dev);fall=torch.zeros(n,dtype=torch.bool,device=dev);slip=fall.clone();slip_streak=torch.zeros(n,dtype=torch.long,device=dev);stable_speed=[];stable_yaw=[]
 for step in range(125):
  p=torch.full((world.env.num_envs,),min(1.,step/75),device=dev);physical=target*minimum_jerk(p)[:,None];world.state.advance(physical,p,DT);set_command(world,physical);base=world.env.observation_manager.compute()["policy"]
  with torch.inference_mode():a=walk(base,gait)
  _,_,done,extras=world.wrapped.step(a);done=done.bool();timeout=extras.get("time_outs",torch.zeros_like(done)).bool();fall|=done[:n]&~timeout[:n];force=world.sensor.data.net_forces_w_history[:n,-1,world.sf,:].norm(dim=-1);contact=force>5;feet=world.robot.data.body_lin_vel_w[:n,world.rf,:2].norm(dim=-1);bad=((feet>.55)&contact).any(1);slip_streak=torch.where(bad,slip_streak+1,torch.zeros_like(slip_streak));slip|=slip_streak>=5
  if step>=75:stable_speed.append(world.robot.data.root_lin_vel_b[:n,:2].detach().cpu());stable_yaw.append(world.robot.data.root_ang_vel_b[:n,2].detach().cpu())
 sv=torch.stack(stable_speed);yv=torch.stack(stable_yaw);acq=[]
 for j,e in enumerate(entries):
  target_j=target[j].cpu();translation_ok=float((sv[:,j]-target_j[:2]).norm(dim=1).mean())<=.18 if e["condition"]["speed"] else float(sv[:,j].norm(dim=1).mean())<=.12;yaw_ok=float((yv[:,j]-target_j[2]).abs().mean())<=.18;acq.append(bool(translation_ok and yaw_ok and not fall[j] and not slip[j]))
 snap=world.snapshot();snap={k:v.detach().cpu() for k,v in snap.items()};payload={"snapshot":snap,"entries":entries,"active":n,"w_move_acquired":acq,"stable_speed":sv,"stable_yaw":yv,"snapshot_hash":sha_bytes(*(v[:n] for v in snap.values()))};path=Path(path) if path else RAW/f"validation_snapshot_batch_{batch_id:02d}.pt";path.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,path);return payload,path

def restore_payload(world,payload):
 s={k:v.to(world.device) for k,v in payload["snapshot"].items()};return world.restore_snapshot(s)

def route_batch(world,payload,route,walk,stop,hold,capture_path=None):
 obs=restore_payload(world,payload);n=payload["active"];entries=payload["entries"];eligible=torch.tensor(payload["w_move_acquired"],device=world.device);dev=world.device;target=command_matrix(entries,world.env.num_envs,dev);world.state.request(torch.full((world.env.num_envs,),int(MotionMode.STAND),device=dev));gait=torch.zeros(world.env.num_envs,device=dev);fall=torch.zeros(n,dtype=torch.bool,device=dev);slip=fall.clone();impact=fall.clone();sat=fall.clone();slip_streak=torch.zeros(n,dtype=torch.long,device=dev);sat_streak=slip_streak.clone();speed=[];yaw=[];support=[];roll_pitch=[];actions=[];safety=[];label_fields={k:[] for k in ("observation_141","physical_command","previous_command","command_delta","time_since_mode_change","ramp_progress","previous_action","motion_mode","previous_mode")} if capture_path else None;streak=torch.zeros(n,dtype=torch.long,device=dev);completion=torch.full((n,),-1,dtype=torch.long,device=dev);jump=torch.zeros(n,device=dev);cos=torch.ones(n,device=dev);jump_groups={g:torch.zeros(n,device=dev) for g in ("legs","waist","torso_arms","hands")};group_ids={"legs":[0,1,3,4,7,8,11,12,15,16,19,20],"waist":[2],"torso_arms":list(range(5,23)),"hands":list(range(23,37))}
 for step in range(200):
  p=torch.full((world.env.num_envs,),min(1.,step/25),device=dev);physical=target*(1-minimum_jerk(p))[:,None];world.state.advance(physical,p,0. if step==0 else DT);set_command(world,physical);base=world.env.observation_manager.compute()["policy"];obs141=world.obs()
  with torch.inference_mode():aw=walk(base,gait);astop=stop.mean(obs141) if hasattr(stop,"mean") else stop(base,gait);ahold=hold.mean(obs141)
  if route=="R0_W_MOVE_ZERO":a=aw
  elif route=="R1_STAGE2Q_STOP":a=astop
  elif route=="R2_S_HOLD_DIRECT":a=ahold
  elif route=="R3_W_MOVE_THEN_S_HOLD":a=torch.where((completion>=0)[:,None],ahold[:n],aw[:n]);a=torch.cat((a,aw[n:]),0)
  elif route=="R5_DEDICATED_OMNI_STOP":a=torch.where((completion>=0)[:,None],ahold[:n],astop[:n]);a=torch.cat((a,astop[n:]),0)
  else:
   pre=aw if step<25 else astop;a=torch.where((completion>=0)[:,None],ahold[:n],pre[:n]);a=torch.cat((a,pre[n:]),0)
  if label_fields is not None:
   label_fields["observation_141"].append(obs141[:n].detach().cpu());label_fields["physical_command"].append(physical[:n].detach().cpu());label_fields["previous_command"].append(world.state.previous_physical_command[:n].detach().cpu());label_fields["command_delta"].append((world.state.physical_command-world.state.previous_physical_command)[:n].detach().cpu());label_fields["time_since_mode_change"].append(world.state.time_since_mode_change_s[:n].detach().cpu());label_fields["ramp_progress"].append(world.state.ramp_progress[:n].detach().cpu());label_fields["previous_action"].append(world.env.action_manager.prev_action[:n].detach().cpu());label_fields["motion_mode"].append(world.state.target_mode[:n].detach().cpu());label_fields["previous_mode"].append(world.state.previous_target_mode[:n].detach().cpu())
  newly_handoff=completion==step-1
  if newly_handoff.any() and route in ("R3_W_MOVE_THEN_S_HOLD","R4_W_MOVE_STAGE2Q_HOLD","R5_DEDICATED_OMNI_STOP"):
   source=aw[:n] if route=="R3_W_MOVE_THEN_S_HOLD" else astop[:n];diff=ahold[:n]-source;jump[newly_handoff]=diff[newly_handoff].norm(dim=1);cos[newly_handoff]=torch.nn.functional.cosine_similarity(ahold[:n][newly_handoff],source[newly_handoff],dim=1)
   for g,ids in group_ids.items():jump_groups[g][newly_handoff]=diff[newly_handoff][:,ids].norm(dim=1)
  actions.append(a[:n].detach().cpu());_,_,done,extras=world.wrapped.step(a);done=done.bool();timeout=extras.get("time_outs",torch.zeros_like(done)).bool();fall|=done[:n]&~timeout[:n];force=world.sensor.data.net_forces_w_history[:n,-1,world.sf,:].norm(dim=-1);contact=force>5;feet=world.robot.data.body_lin_vel_w[:n,world.rf,:2].norm(dim=-1);bad=((feet>.55)&contact).any(1);slip_streak=torch.where(bad,slip_streak+1,torch.zeros_like(slip_streak));slip|=slip_streak>=5;impact|=force.amax(1)>3500;ratio=world.robot.data.joint_vel[:n].abs().div(world.limits[:n].clamp_min(1e-6)).amax(1);sat_streak=torch.where(ratio>.95,sat_streak+1,torch.zeros_like(sat_streak));sat|=sat_streak>=5;s=world.robot.data.root_lin_vel_b[:n,:2].norm(dim=1);y=world.robot.data.root_ang_vel_b[:n,2].abs();good=(s<=.08)&(y<=.08);streak=torch.where(good,streak+1,torch.zeros_like(streak));new=(completion<0)&(streak>=25)&((step-24)<75);completion[new]=step;speed.append(s.detach().cpu());yaw.append(y.detach().cpu());support.append(contact.sum(1).detach().cpu());safety.append(torch.stack((fall,slip,impact,sat),dim=1).detach().cpu());grav=world.robot.data.projected_gravity_b[:n];roll_pitch.append(torch.acos((-grav[:,2]).clamp(-1,1)).detach().cpu())
 speed=torch.stack(speed);yaw=torch.stack(yaw);support=torch.stack(support);safehist=torch.stack(safety);rp=torch.stack(roll_pitch);actions=torch.stack(actions);rows=[]
 for j,e in enumerate(entries):
  comp=int(completion[j]);safe_at_comp=comp>=0 and not bool(safehist[comp,j].any());stop_pass=bool(eligible[j] and comp>=0 and safe_at_comp);start=comp+1;end=start+100;hold_ok=False
  if stop_pass and end<=200:
   hs=speed[start:end,j];hy=yaw[start:end,j];hold_ok=bool(hs.mean()<=.08 and hy.mean()<=.08 and torch.quantile(hs,.95)<=.12 and torch.quantile(hy,.95)<=.12 and not safehist[end-1,j].any())
  if not eligible[j]:cause="W_MOVE_START_ACQUISITION_FAIL"
  elif fall[j]:cause="FALL_DURING_BRAKING"
  elif slip[j]:cause="SLIP_DURING_BRAKING"
  elif comp<0:
   cause="YAW_NOT_DAMPED" if float(yaw[:100,j].mean())>.08 else "LATERAL_VELOCITY_NOT_DAMPED" if abs(math.sin(math.radians(e["condition"]["direction_deg"])))>.5 else "NO_DECELERATION"
  elif not hold_ok:cause="STAND_BASIN_NOT_REACHED"
  else:cause=None
  rows.append({"route":route,"condition_id":e["condition"]["condition_id"],"kind":e["condition"]["kind"],"direction_deg":e["condition"]["direction_deg"],"command_yaw":e["condition"]["yaw"],"recipe_id":e["recipe_id"],"snapshot_id":e["snapshot_id"],"w_move_start_acquired":bool(eligible[j]),"stop_acquisition":stop_pass,"acquisition_step":None if comp<0 else comp-24,"confirmation_completion_step":None if comp<0 else comp,"acquisition_time_s":None if comp<0 else (comp-23)*DT,"stand_after_stop":hold_ok if stop_pass else None,"joint_success":bool(stop_pass and hold_ok),"fall":bool(fall[j]),"dangerous_slip":bool(slip[j]),"impact":bool(impact[j]),"saturation":bool(sat[j]),"handoff_action_l2":float(jump[j]),"handoff_action_cosine":float(cos[j]),"jump_legs":float(jump_groups["legs"][j]),"jump_waist":float(jump_groups["waist"][j]),"jump_torso_arms":float(jump_groups["torso_arms"][j]),"jump_hands":float(jump_groups["hands"][j]),"root_state_discontinuity":0.,"contact_discontinuity":0.,"failure_cause":cause,"failure_onset_step":0 if cause and cause.startswith("FALL") else None,"speed_trajectory":speed[:,j].tolist(),"yaw_trajectory":yaw[:,j].tolist(),"contact_sequence":support[:,j].tolist(),"roll_pitch_trajectory":rp[:,j].tolist(),"action_hash":sha_bytes(actions[:,j])})
 if capture_path:
  stacked={k:torch.stack(v) for k,v in label_fields.items()};flat={k:[] for k in stacked};flat["action_37"]=[];meta={k:[] for k in ("recipe_id","split","condition_id","control_step","teacher_role_id","checkpoint_id")}
  for j,e in enumerate(entries):
   if not eligible[j] or int(completion[j])<0:continue
   last=min(199,int(completion[j])+25)
   for k in stacked:flat[k].append(stacked[k][:last+1,j])
   flat["action_37"].append(actions[:last+1,j]);steps=torch.arange(last+1);meta["recipe_id"].append(torch.full_like(steps,e["recipe_id"]));meta["condition_id"].append(torch.full_like(steps,e["condition"]["condition_id"]));meta["control_step"].append(steps);meta["split"].append(torch.full_like(steps,{"train":0,"validation":1,"held-out":2}[e["split"]]));meta["teacher_role_id"].append(torch.where(steps<=int(completion[j]),torch.zeros_like(steps),torch.ones_like(steps)));meta["checkpoint_id"].append(torch.where(steps<25,torch.zeros_like(steps),torch.where(steps<=int(completion[j]),torch.ones_like(steps),torch.full_like(steps,2))))
  labels={k:torch.cat(v) for k,v in flat.items()};labels.update({k:torch.cat(v) for k,v in meta.items()});capture_path=Path(capture_path);capture_path.parent.mkdir(parents=True,exist_ok=True);torch.save(labels,capture_path)
 return rows

def summarize(rows,route):
 valid=[r for r in rows if r["w_move_start_acquired"]];conds=[]
 for cid in range(34):
  x=[r for r in valid if r["condition_id"]==cid];st=[r for r in x if r["stop_acquisition"]];conds.append({"condition_id":cid,"kind":x[0]["kind"] if x else None,"direction_deg":x[0]["direction_deg"] if x else None,"command_yaw":x[0]["command_yaw"] if x else None,"denominator":len(x),"stop_acquisition":rate([r["stop_acquisition"] for r in x]),"conditional_stand_after_stop":rate([r["stand_after_stop"] for r in st]),"joint_success":rate([r["joint_success"] for r in x]),"fall":rate([r["fall"] for r in x]),"dangerous_slip":rate([r["dangerous_slip"] for r in x]),"impact":rate([r["impact"] for r in x]),"saturation":rate([r["saturation"] for r in x])})
 st=[r for r in valid if r["stop_acquisition"]];jumps=[r["handoff_action_l2"] for r in st];coss=[r["handoff_action_cosine"] for r in st];q=lambda x,p:float(torch.quantile(torch.tensor(x),p)) if x else None;agg={"route":route,"snapshots_total":len(rows),"w_move_start_acquired":len(valid),"w_move_start_acquisition_rate":len(valid)/len(rows),"conditions_evaluated":len({r["condition_id"] for r in valid}),"stop_acquisition":rate([r["stop_acquisition"] for r in valid]),"conditional_stand_after_stop":rate([r["stand_after_stop"] for r in st]),"joint_success":rate([r["joint_success"] for r in valid]),"fall":rate([r["fall"] for r in valid]),"dangerous_slip":rate([r["dangerous_slip"] for r in valid]),"impact":rate([r["impact"] for r in valid]),"saturation":rate([r["saturation"] for r in valid]),"minimum_condition_joint_success":min((c["joint_success"] for c in conds),default=0),"handoff_action_l2_p95":q(jumps,.95),"handoff_action_cosine_p05":q(coss,.05),"conditions":conds}
 agg["eligible"]=agg["conditions_evaluated"]==34 and all(c["stop_acquisition"]>=.95 and c["conditional_stand_after_stop"]>=.95 and c["joint_success"]>=.90 and c["fall"]<=.02 and c["dangerous_slip"]<=.05 and c["impact"]<=.05 and c["saturation"]<=.05 for c in conds) and agg["minimum_condition_joint_success"]>=.80 and (agg["handoff_action_l2_p95"] or 0)<=.5 and (agg["handoff_action_cosine_p05"] or 1)>=.98
 return agg

def main():
 p=argparse.ArgumentParser();add_launcher_args(p);args,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0],*hydra];RAW.mkdir(parents=True,exist_ok=True);cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=476;cfg.seed=20279001;cfg.episode_length_s=20.;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None
 if args.device:cfg.sim.device=agent.device=args.device
 resets=d3.load_resets();severity=torch.zeros(680)
 with launch_simulation(cfg,args):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=d3.StandWorld(wrapped,resets,severity);walk=FrozenGaitActor(d3.WMOVE).to(world.device).eval();stop=FrozenGaitActor(d3.P1).to(world.device).eval();hold=d3.initialize("P0_STAND_PARENT",world.device)[0];conds=conditions();entries=[]
  recipes=d3.VALIDATION[:100]
  for c in conds:
   for i,r in enumerate(recipes):entries.append({"snapshot_id":f"validation_c{c['condition_id']:02d}_e{i:03d}","split":"validation","recipe_id":r,"condition":c})
  payloads=[];manifest=[]
  for b,start in enumerate(range(0,len(entries),476)):
   payload,path=generate_batch(world,walk,entries[start:start+476],b);payloads.append(payload);manifest.append({"batch":b,"path":str(path.relative_to(REPO)).replace("\\","/"),"active":payload["active"],"snapshot_hash":payload["snapshot_hash"],"w_move_acquired":sum(payload["w_move_acquired"])})
  all_rows=[];summaries=[]
  for route in ROUTES:
   rows=[]
   for payload in payloads:rows.extend(route_batch(world,payload,route,walk,stop,hold))
   all_rows.extend(rows);summaries.append(summarize(rows,route));dump(RAW/f"{route}.json",{"summary":summaries[-1],"rows":rows})
  dump(RAW/"audit_results.json",{"conditions":conds,"snapshot_manifest":manifest,"routes":summaries});wrapped.close()

if __name__=="__main__":main()
