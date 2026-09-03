"""Search weight-shift plus lead-foot swing sequences for a 5 cm obstacle."""

from __future__ import annotations

import argparse, csv, json, sys
from importlib import metadata
from pathlib import Path
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT.parents[2]
sys.path[:0]=[str(ROOT/"src"),str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src")]
import isaaclab_tasks  # noqa:E402,F401
import g1_flat_run.tasks  # noqa:E402,F401
import g1_command_skills.tasks  # noqa:E402,F401
from isaaclab.utils.math import quat_apply  # noqa:E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa:E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa:E402
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli  # noqa:E402

p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--candidates-per-side",type=int,default=512);p.add_argument("--seed",type=int,default=20260723);add_launcher_args(p);args,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0]]+hydra
SOLE=(0.04321213238651294,0.0,-0.025807180037281774);TOE=(0.06383880963290349,0.0,-0.025807180037281774)
J={"hip_pitch":(0,1),"hip_roll":(3,4),"knee":(11,12),"ankle_pitch":(15,16),"ankle_roll":(19,20)}
def mj(x): x=max(0.,min(1.,x));return x**3*(10-15*x+6*x*x)
def main():
 out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True);g=torch.Generator().manual_seed(args.seed);c=[]
 for side in ("left","right"):
  for i in range(args.candidates_per_side):
   r=torch.rand(7,generator=g).tolist()
   weight = ({"lead_hip_roll":0.1665648222,"support_hip_roll":0.3186047077,"lead_ankle_roll":0.4256471992,"support_ankle_roll":-0.4985398054,"left_shoulder_roll":1.0013181209,"right_shoulder_roll":-0.4743543148,"left_shoulder_pitch":-0.0628775597,"right_shoulder_pitch":0.0057731628}
             if side=="left" else {"lead_hip_roll":-0.1614698172,"support_hip_roll":-0.4249010086,"lead_ankle_roll":-0.1783357859,"support_ankle_roll":0.3713575006,"left_shoulder_roll":-0.5788853645,"right_shoulder_roll":-0.0400099754,"left_shoulder_pitch":-0.3689019203,"right_shoulder_pitch":-0.6599188805})
   c.append({"side":side,"candidate":i,**weight,
    "lift_hip_pitch":.05+.25*r[0],"lift_knee":.10+1.30*r[1],"lift_ankle_pitch":-1.50+2.50*r[2],
    "swing_hip_pitch":-1.10+.90*r[3],"support_hip_pitch":-.20+.40*r[4],"support_ankle_pitch":-.30+.60*r[5],
    "swing_knee_delta":-.20+.40*r[6]})
 n=len(c);cfg,acfg=resolve_task_config("Isaac-Motion-Flat-G1-Command-StepOverAudit-Eval-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=n;cfg.seed=args.seed
 if args.device is not None:cfg.sim.device=args.device
 with launch_simulation(cfg,args):
  raw=gym.make("Isaac-Motion-Flat-G1-Command-StepOverAudit-Eval-v0",cfg=cfg);w=RslRlVecEnvWrapper(raw,clip_actions=acfg.clip_actions);e=raw.unwrapped;acfg.device=e.device;acfg=handle_deprecated_rsl_rl_cfg(acfg,metadata.version("rsl-rl-lib"));run=OnPolicyRunner(w,acfg.to_dict(),log_dir=None,device=acfg.device);run.load(str(Path(args.checkpoint).resolve(strict=True)),load_cfg={"actor":True,"critic":False,"optimizer":False,"iteration":False,"rnd":False});a=run.alg.actor;robot=e.scene["robot"];contact=e.scene.sensors["contact_forces"];obs_contact=e.scene.sensors["step_obstacle_contact"]
  fids,fn=robot.find_bodies(["left_ankle_roll_link","right_ankle_roll_link"],preserve_order=True);sids=[contact.body_names.index(x) for x in fn];allj,_=robot.find_joints(".*")
  weight=torch.zeros(n,37,device=e.device);lift=torch.zeros_like(weight);swing=torch.zeros_like(weight);lead_index=[]
  for k,x in enumerate(c):
   l=0 if x["side"]=="left" else 1;s=1-l;lead_index.append(l)
   weight[k,J["hip_roll"][l]]=x["lead_hip_roll"];weight[k,J["hip_roll"][s]]=x["support_hip_roll"];weight[k,J["ankle_roll"][l]]=x["lead_ankle_roll"];weight[k,J["ankle_roll"][s]]=x["support_ankle_roll"]
   weight[k,9]=x["left_shoulder_roll"];weight[k,10]=x["right_shoulder_roll"];weight[k,5]=x["left_shoulder_pitch"];weight[k,6]=x["right_shoulder_pitch"]
   lift[k]=weight[k];lift[k,J["hip_pitch"][l]]=x["lift_hip_pitch"];lift[k,J["knee"][l]]=x["lift_knee"];lift[k,J["ankle_pitch"][l]]=x["lift_ankle_pitch"];lift[k,J["hip_pitch"][s]]=x["support_hip_pitch"];lift[k,J["ankle_pitch"][s]]=x["support_ankle_pitch"]
   swing[k]=lift[k];swing[k,J["hip_pitch"][l]]=x["swing_hip_pitch"];swing[k,J["knee"][l]]=x["lift_knee"]+x["swing_knee_delta"]
  lead=torch.tensor(lead_index,device=e.device);support=1-lead;w.reset();dt=float(e.step_dt);segments=[round(x/dt) for x in (1.5,1.0,.3,1.0,.3,1.2,.5)];ends=[];z=0
  for x in segments:z+=x;ends.append(z)
  active=torch.ones(n,dtype=torch.bool,device=e.device);lead_released=torch.zeros_like(active);collision=torch.zeros_like(active);vs=torch.zeros(n,device=e.device);ts=torch.zeros(n,device=e.device);samples=0
  support_loss_run=torch.zeros(n,device=e.device);support_loss_max=support_loss_run.clone();air_run=support_loss_run.clone();air_max=support_loss_run.clone()
  initial=None;maxsole=torch.full((n,),-1e9,device=e.device);maxtoe=maxsole.clone();maxx=torch.full((n,),-1e9,device=e.device)
  for step in range(ends[-1]):
   ob=w.get_observations();standing=a.diagnostic_components(ob)["standing_base_action"]
   if step<ends[0]:off=weight*0
   elif step<ends[1]:off=weight*mj((step-ends[0]+1)/segments[1])
   elif step<ends[2]:off=weight
   elif step<ends[3]:
    b=mj((step-ends[2]+1)/segments[3]);off=weight+(lift-weight)*b
   elif step<ends[4]:off=lift
   elif step<ends[5]:
    b=mj((step-ends[4]+1)/segments[5]);off=lift+(swing-lift)*b
   else:off=swing
   _,_,done,_=w.step(standing+off);active&=~done.bool();pos=robot.data.body_pos_w.torch[:,fids];quat=robot.data.body_quat_w.torch[:,fids];sole=pos+quat_apply(quat.reshape(-1,4),torch.tensor(SOLE,device=e.device).expand(n,2,3).reshape(-1,3)).reshape(n,2,3);toe=pos+quat_apply(quat.reshape(-1,4),torch.tensor(TOE,device=e.device).expand(n,2,3).reshape(-1,3)).reshape(n,2,3)
   if step==ends[0]-1:initial=sole.clone()
   if step>=ends[2]:
    samples+=1;lc=sole.gather(1,lead[:,None,None].expand(-1,1,3)).squeeze(1);lt=toe.gather(1,lead[:,None,None].expand(-1,1,3)).squeeze(1);maxsole=torch.maximum(maxsole,lc[:,2]);maxtoe=torch.maximum(maxtoe,lt[:,2]);maxx=torch.maximum(maxx,lc[:,0]);forces=contact.data.net_forces_w_history.torch[:,:,sids,:].norm(dim=-1).amax(dim=1)>5.;support_now=forces.gather(1,support[:,None]).squeeze(1);air_now=~forces.any(dim=1);support_loss_run=torch.where(support_now,torch.zeros_like(support_loss_run),support_loss_run+1);support_loss_max=torch.maximum(support_loss_max,support_loss_run);air_run=torch.where(air_now,air_run+1,torch.zeros_like(air_run));air_max=torch.maximum(air_max,air_run);lead_released|=~forces.gather(1,lead[:,None]).squeeze(1);collision|=obs_contact.data.net_forces_w_history.torch.norm(dim=-1).amax(dim=(1,2))>5.;vr=robot.data.joint_vel.torch[:,allj].abs()/robot.data.joint_vel_limits.torch[:,allj].abs().clamp_min(1e-6);tr=robot.data.applied_torque.torch[:,allj].abs()/robot.data.joint_effort_limits.torch[:,allj].abs().clamp_min(1e-6);vs+=(vr>=.95).any(dim=1);ts+=(tr>=.95).any(dim=1)
  grav=robot.data.projected_gravity_b.torch;tilt=torch.linalg.vector_norm(grav[:,:2],dim=1);rows=[]
  for k,x in enumerate(c):
   l=lead_index[k];row=dict(x);row.update({"sole_lift_m":float(maxsole[k]-initial[k,l,2]),"toe_lift_m":float(maxtoe[k]-initial[k,l,2]),"sole_forward_m":float(maxx[k]-initial[k,l,0]),"maximum_support_loss_duration_s":float(support_loss_max[k]*dt),"lead_released":bool(lead_released[k]),"maximum_both_feet_airborne_duration_s":float(air_max[k]*dt),"obstacle_collision":bool(collision[k]),"fall":not bool(active[k]),"tilt_rad":float(tilt[k]),"velocity_saturation_fraction":float(vs[k]/samples),"torque_saturation_fraction":float(ts[k]/samples)});row["viable_swing"]=bool(active[k] and support_loss_max[k]*dt<=.10 and lead_released[k] and air_max[k]*dt<=.10 and not collision[k] and tilt[k]<.25 and vs[k]/samples<=.05 and ts[k]/samples<=.05 and row["sole_lift_m"]>=.075 and row["sole_forward_m"]>=.36);rows.append(row)
  fields=list(dict.fromkeys(k for r in rows for k in r));f=open(out/"sequences.csv","w",newline="",encoding="utf-8");dw=csv.DictWriter(f,fieldnames=fields);dw.writeheader();dw.writerows(rows);f.close();viable=[r for r in rows if r["viable_swing"]];viable.sort(key=lambda r:(r["sole_forward_m"],r["sole_lift_m"]),reverse=True);summary={"candidate_count":n,"viable_count":len(viable),"requirements":{"sole_lift_m":.075,"sole_forward_m":.36},"best":viable[:20]};(out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8");print(json.dumps({"viable_count":len(viable),"best":viable[:3]},indent=2));raw.close()
if __name__=="__main__":main()
