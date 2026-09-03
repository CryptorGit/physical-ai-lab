"""Deterministic yaw-conditioned W1B evaluator with body-frame metrics."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys
from collections import defaultdict
from pathlib import Path
import gymnasium as gym
import torch

HERE=Path(__file__).resolve();REPO=HERE.parents[4]
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_yaw_conditioned_omnidirectional_walk"
sys.path.insert(0,str(HERE.parent.parent/"src"))
import isaaclab_tasks,g1_omnidirectional.tasks
from g1_omnidirectional.policy import FrozenGaitActor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli

p=argparse.ArgumentParser();p.add_argument("--mode",required=True,choices=("parent","capability","zero","pure","moving","independence","envelope","path","random"));p.add_argument("--checkpoint",required=True);p.add_argument("--tag",required=True);add_launcher_args(p)
a,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0],*hydra]
def static(name,speed,deg,yaw,episodes,kind="moving",duration=8):
 r=math.radians(deg);return {"name":name,"speed":speed,"direction_deg":deg,"yaw_cmd":yaw,"episodes":episodes,"duration":duration,"kind":kind,"vx":speed*math.cos(r),"vy":speed*math.sin(r)}
def specs():
 if a.mode=="parent":
  s=[static(f"PURE_Y{y:+.2f}",0,0,y,30,"pure") for y in (-.6,-.45,-.3,-.2,.2,.3,.45,.6)]
  s += [static(f"FWD_S{v:.1f}_Y{y:+.1f}",v,0,y,30,"zero" if y==0 else "moving") for v in (.3,.6) for y in (-.6,-.3,0,.3,.6)]
  s += [static(f"LAT_D{d:.1f}_Y{y:+.1f}",.3,d,y,30,"zero" if y==0 else "moving") for d in (90,270) for y in (-.3,0,.3)]
  return s
 if a.mode=="capability":
  s=[static(f"ZERO_D{d:05.1f}",.3,d,0,20,"zero") for d in (i*22.5 for i in range(16))]
  s += [static("FWD_0P6",.6,0,0,20,"zero"),static("FWD_1P2",1.2,0,0,20,"zero")]
  s += [static(f"PURE_Y{y:+.1f}",0,0,y,20,"pure") for y in (-.3,.3)]
  s += [static(f"FWD_Y{y:+.1f}",.3,0,y,20,"moving") for y in (-.3,.3)]
  s += [static(f"LAT_D{d:.0f}_Y{y:+.1f}",.3,d,y,20,"moving") for d in (90,270) for y in (-.3,.3)]
  return s
 if a.mode=="zero":
  return [static(f"ZERO_D{d:05.1f}",.3,d,0,50,"zero") for d in (i*22.5 for i in range(16))]+[static("FWD_0P6",.6,0,0,50,"zero"),static("FWD_1P2",1.2,0,0,50,"zero")]
 if a.mode=="pure":
  return [static(f"PURE_Y{y:+.1f}",0,0,y,100 if abs(y)==.3 else 50,"pure") for y in (-.3,.3,-.6,.6)]
 if a.mode=="moving":
  return [static(f"MOVE_D{d:05.1f}_Y{y:+.1f}",.3,d,y,50,"zero" if y==0 else "moving") for d in range(0,360,45) for y in (-.3,0,.3)]
 if a.mode=="independence":
  pairs=[(270,.3),(270,-.3),(90,.3),(90,-.3),(45,-.3),(315,.3),(135,-.3),(225,.3),(180,.3),(180,-.3)]
  return [static(f"IND_D{d:05.1f}_Y{y:+.1f}",.3,d,y,50,"moving") for d,y in pairs]
 if a.mode=="envelope":
  q=[]
  for v in (.6,1.): q += [static(f"FWD_S{v:.1f}_Y{y:+.1f}",v,0,y,30,"moving") for y in (-.5,-.3,.3,.5)]
  q += [static(f"FD_D{d:.0f}_Y{y:+.1f}",.6,d,y,30,"moving") for d in (45,315) for y in (-.3,.3)]
  q += [static(f"LR_D{d:.0f}_S{v:.1f}_Y{y:+.1f}",v,d,y,30,"moving") for d in (90,135,180,225,270) for v in (.3,.5) for y in (-.3,.3)]
  return q
 if a.mode=="path":
  return [{"name":"CIRCLE_LEFT","episodes":50,"duration":16,"kind":"path","shape":"circle","sign":1},{"name":"CIRCLE_RIGHT","episodes":50,"duration":16,"kind":"path","shape":"circle","sign":-1},{"name":"S_CURVE","episodes":50,"duration":18,"kind":"path","shape":"s","sign":1},{"name":"STRAFE_LEFT_RIGHTTURN","episodes":50,"duration":12,"kind":"path","shape":"strafe","sign":1},{"name":"STRAFE_RIGHT_LEFTTURN","episodes":50,"duration":12,"kind":"path","shape":"strafe","sign":-1}]
 return [{"name":"RANDOM_60S","episodes":30,"duration":60,"kind":"random"}]
def cmd(item,t,episode):
 if item["kind"] not in ("path","random"):return item["vx"],item["vy"],item["yaw_cmd"]
 if item["kind"]=="random":
  segment=min(int(t//4),14);g=torch.Generator().manual_seed(20274021+episode);ang=torch.rand(15,generator=g)*2*math.pi;sp=torch.rand(15,generator=g)*.4;y=torch.rand(15,generator=g)*.8-.4
  return float(sp[segment]*torch.cos(ang[segment])),float(sp[segment]*torch.sin(ang[segment])),float(y[segment])
 if item["shape"]=="circle":return .4,0,.3*item["sign"]
 if item["shape"]=="strafe":return 0,.3*item["sign"],-.3*item["sign"]
 # minimum-jerk proxy over 2 s between +,-,+ yaw plateaus
 targets=(.3,-.3,.3);segment=min(int(t//6),2);local=t-segment*6;prev=targets[max(segment-1,0)];target=targets[segment];u=min(local/2,1);r=u**3*(10-15*u+6*u*u);return .4,0,prev+(target-prev)*r
def main():
 cfg,ac=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point")
 spec=specs();total=sum(x["episodes"] for x in spec);cfg.scene.num_envs=total;cfg.episode_length_s=max(x["duration"] for x in spec)+2;cfg.seed=20274021
 if a.device:cfg.sim.device=ac.device=a.device
 with launch_simulation(cfg,a):
  w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=ac.clip_actions);e=w.unwrapped;dev=e.device;actor=FrozenGaitActor(a.checkpoint).to(dev).eval();robot=e.scene["robot"];sensor=e.scene.sensors["contact_forces"];term=e.command_manager.get_term("base_velocity");term.external_override_enabled=True
  feet=[i for i,n in enumerate(sensor.body_names) if "ankle_roll" in n];rfeet=[robot.body_names.index(sensor.body_names[i]) for i in feet]
  ci=[];epi=[]
  for i,x in enumerate(spec):
   ci += [i]*x["episodes"];epi += list(range(x["episodes"]))
  ids=torch.tensor(ci,device=dev);obs,_=w.reset();obs=obs.to(dev);n=total;active=torch.ones(n,dtype=torch.bool,device=dev)
  fall=torch.zeros_like(active);tiltbad=fall.clone();slipbad=fall.clone();impact=fall.clone();satbad=fall.clone();slipst=torch.zeros(n,dtype=torch.long,device=dev);satst=slipst.clone()
  sums={k:torch.zeros(n,device=dev) for k in ("vx","vy","speed","vec","dir","yaw","yawerr","drift","flight","slip","roll","pitch","left","right")};steps=torch.zeros(n,device=dev)
  maxsteps=round(max(x["duration"] for x in spec)/e.step_dt)
  for st in range(maxsteps):
   t=st*e.step_dt;vx=torch.zeros(n,device=dev);vy=vx.clone();yc=vx.clone();valid=torch.zeros(n,dtype=torch.bool,device=dev)
   for i,x in enumerate(spec):
    m=torch.where(ids==i)[0]
    if t>=x["duration"]:continue
    valid[m]=True
    for j in m.tolist():vx[j],vy[j],yc[j]=cmd(x,t,epi[j])
   term.external_override[:,0]=vx;term.external_override[:,1]=vy;term.external_override[:,2]=yc
   if st==0:term._update_command();obs=w.get_observations().to(dev)
   with torch.inference_mode():act=actor(obs["policy"],torch.zeros(n,device=dev))
   obs,_,done,extra=w.step(act);obs=obs.to(dev);measure=active&valid;timeout=extra.get("time_outs",torch.zeros_like(done)).bool();fall|=done.bool()&~timeout&measure
   av=robot.data.root_lin_vel_b[:,:2];ay=robot.data.root_ang_vel_b[:,2];force=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1);contact=force>5;fs=torch.linalg.vector_norm(robot.data.body_lin_vel_w[:,rfeet,:2],dim=-1);sl=((fs>.55)&contact).any(-1);slipst=torch.where(sl,slipst+1,torch.zeros_like(slipst));slipbad|=(slipst>=5)&measure;impact|=(force.amax(-1)>3500)&measure
   g=robot.data.projected_gravity_b;roll=torch.atan2(g[:,1].abs(),g[:,2].abs().clamp_min(1e-6));pitch=torch.atan2(g[:,0].abs(),g[:,2].abs().clamp_min(1e-6));tiltbad|=(torch.maximum(roll,pitch)>.8)&measure
   lim=robot.data.joint_vel_limits;lim=lim[...,1].abs() if lim.ndim==3 else lim;sat=robot.data.joint_vel.abs().div(lim.clamp_min(1e-6)).amax(-1)>.95;satst=torch.where(sat,satst+1,torch.zeros_like(satst));satbad|=(satst>=5)&measure
   vec=torch.linalg.vector_norm(av-torch.stack((vx,vy),1),dim=-1);ca=torch.atan2(vy,vx);aa=torch.atan2(av[:,1],av[:,0]);dire=torch.atan2(torch.sin(aa-ca),torch.cos(aa-ca)).abs()*180/math.pi
   vals={"vx":av[:,0],"vy":av[:,1],"speed":torch.linalg.vector_norm(av,dim=-1),"vec":vec,"dir":dire,"yaw":ay,"yawerr":(ay-yc).abs(),"drift":torch.linalg.vector_norm(av,dim=-1),"flight":(contact.sum(-1)==0).float(),"slip":sl.float(),"roll":roll,"pitch":pitch,"left":contact[:,0].float(),"right":contact[:,1].float()}
   for k,v in vals.items():sums[k]+=torch.where(measure,v,0)
   steps+=measure.float();active&=~(fall|impact|satbad|(torch.maximum(roll,pitch)>.8))
  erows=[]
  for j in range(n):
   den=max(float(steps[j]),1);m={k:float(v[j]/den) for k,v in sums.items()};item=spec[ci[j]];yawcmd=cmd(item,0,epi[j])[2];gait="FALL" if fall[j] else ("TURN_IN_PLACE" if item["kind"]=="pure" and m["flight"]<.1 else ("TURNING_WALK" if abs(yawcmd)>.01 and m["flight"]<.1 else ("TRANSLATION_ONLY" if m["flight"]<.1 else "UNTRACKED")))
   safe=not bool(fall[j] or slipbad[j] or impact[j] or satbad[j]);sign=(m["yaw"]*yawcmd)>0 if abs(yawcmd)>.01 else abs(m["yaw"])<=.2
   transok=m["vec"]<=.25 and (item.get("speed",math.hypot(cmd(item,0,epi[j])[0],cmd(item,0,epi[j])[1]))<.05 or m["dir"]<=25)
   yawok=(m["yawerr"]<=.15 if item["kind"]=="pure" else m["yawerr"]<=.20) and sign
   if item["kind"]=="pure":success=safe and yawok and m["speed"]<=.12
   elif item["kind"]=="zero":success=safe and m["vec"]<=.20 and m["dir"]<=20 and abs(m["yaw"])<=.2 and m["flight"]<.1
   else:success=safe and transok and yawok and m["flight"]<.1
   erows.append({"condition":item["name"],"episode":epi[j],"kind":item["kind"],"direction_deg":item.get("direction_deg"),"commanded_speed_mps":item.get("speed"),"yaw_cmd":yawcmd,"actual_vx_body":m["vx"],"actual_vy_body":m["vy"],"actual_speed_mps":m["speed"],"actual_yaw_rate":m["yaw"],"vector_velocity_mae":m["vec"],"direction_error_deg":m["dir"],"yaw_rate_mae":m["yawerr"],"yaw_sign_correct":bool(sign),"translation_correct":bool(transok),"yaw_correct":bool(yawok),"both_correct":bool(transok and yawok),"turning_gait":gait,"success":bool(success),"fall":bool(fall[j]),"excessive_tilt":bool(tiltbad[j]),"dangerous_slip":bool(slipbad[j]),"impact_failure":bool(impact[j]),"long_dwell_saturation":bool(satbad[j]),"foot_slip_fraction":m["slip"],"base_roll_abs_mean":m["roll"],"base_pitch_abs_mean":m["pitch"],"left_contact_fraction":m["left"],"right_contact_fraction":m["right"]})
  grouped=defaultdict(list)
  for r in erows:grouped[r["condition"]].append(r)
  rows=[]
  for name,v in grouped.items():
   b={k:v[0].get(k) for k in ("condition","kind","direction_deg","commanded_speed_mps","yaw_cmd")};b["episodes"]=len(v)
   for k in ("actual_vx_body","actual_vy_body","actual_speed_mps","actual_yaw_rate","vector_velocity_mae","direction_error_deg","yaw_rate_mae","foot_slip_fraction","base_roll_abs_mean","base_pitch_abs_mean","left_contact_fraction","right_contact_fraction"):b[k]=sum(x[k] for x in v)/len(v)
   for k in ("success","fall","excessive_tilt","dangerous_slip","impact_failure","long_dwell_saturation","yaw_sign_correct","translation_correct","yaw_correct","both_correct"):b[k+"_rate"]=sum(bool(x[k]) for x in v)/len(v)
   b["gate_pass"]=b["success_rate"]>=.9 and b["fall_rate"]<=.05;rows.append(b)
  payload={"mode":a.mode,"tag":a.tag,"checkpoint":str(Path(a.checkpoint).resolve()),"checkpoint_sha256":hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),"seed":20274021,"deterministic":True,"rows":rows,"episode_rows":erows}
  stem=f"{a.mode}_{a.tag}";(OUT/f"_raw_{stem}.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
  fields=list(rows[0]);f=(OUT/f"_raw_{stem}.csv").open("w",newline="",encoding="utf-8");wr=csv.DictWriter(f,fields);wr.writeheader();wr.writerows(rows);f.close();w.close()
if __name__=="__main__":main()
