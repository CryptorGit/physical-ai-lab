"""Search quasi-static support-side weight-shift poses, including arm counterbalance."""
from __future__ import annotations
import argparse,csv,json,sys
from importlib import metadata
from pathlib import Path
import gymnasium as gym,torch
from rsl_rl.runners import OnPolicyRunner
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[2];sys.path[:0]=[str(ROOT/"src"),str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src")]
import isaaclab_tasks,g1_flat_run.tasks,g1_command_skills.tasks  # noqa:E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa:E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa:E402
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli  # noqa:E402
p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--per-side",type=int,default=512);p.add_argument("--seed",type=int,default=20260724);add_launcher_args(p);args,h=setup_preset_cli(p);sys.argv=[sys.argv[0]]+h
def mj(x):x=max(0.,min(1.,x));return x**3*(10-15*x+6*x*x)
def main():
 out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True);g=torch.Generator().manual_seed(args.seed);rows=[]
 for side in ("left","right"):
  for i in range(args.per_side):
   r=(torch.rand(8,generator=g)*2-1).tolist();rows.append({"side":side,"candidate":i,"lead_hip_roll":.5*r[0],"support_hip_roll":.5*r[1],"lead_ankle_roll":.5*r[2],"support_ankle_roll":.5*r[3],"left_shoulder_roll":1.2*r[4],"right_shoulder_roll":1.2*r[5],"left_shoulder_pitch":.8*r[6],"right_shoulder_pitch":.8*r[7]})
 n=len(rows);cfg,ac=resolve_task_config("Isaac-Motion-Flat-G1-Command-StepOverAudit-Eval-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=n;cfg.seed=args.seed
 with launch_simulation(cfg,args):
  raw=gym.make("Isaac-Motion-Flat-G1-Command-StepOverAudit-Eval-v0",cfg=cfg);w=RslRlVecEnvWrapper(raw,clip_actions=ac.clip_actions);e=raw.unwrapped;ac.device=e.device;ac=handle_deprecated_rsl_rl_cfg(ac,metadata.version("rsl-rl-lib"));run=OnPolicyRunner(w,ac.to_dict(),log_dir=None,device=ac.device);run.load(str(Path(args.checkpoint).resolve(strict=True)),load_cfg={"actor":True,"critic":False,"optimizer":False,"iteration":False,"rnd":False});actor=run.alg.actor;robot=e.scene["robot"];contact=e.scene.sensors["contact_forces"];fids,fn=robot.find_bodies(["left_ankle_roll_link","right_ankle_roll_link"],preserve_order=True);sids=[contact.body_names.index(x) for x in fn]
  off=torch.zeros(n,37,device=e.device);lead=[]
  for k,x in enumerate(rows):
   l=0 if x["side"]=="left" else 1;s=1-l;lead.append(l);off[k,(3,4)[l]]=x["lead_hip_roll"];off[k,(3,4)[s]]=x["support_hip_roll"];off[k,(19,20)[l]]=x["lead_ankle_roll"];off[k,(19,20)[s]]=x["support_ankle_roll"];off[k,9]=x["left_shoulder_roll"];off[k,10]=x["right_shoulder_roll"];off[k,5]=x["left_shoulder_pitch"];off[k,6]=x["right_shoulder_pitch"]
  lead=torch.tensor(lead,device=e.device);support=1-lead;w.reset();dt=float(e.step_dt);settle=round(1.5/dt);ramp=round(1/dt);hold=round(.8/dt);active=torch.ones(n,dtype=torch.bool,device=e.device);force_sum=torch.zeros(n,2,device=e.device);contact_sum=force_sum.clone();tilt_max=torch.zeros(n,device=e.device)
  for step in range(settle+ramp+hold):
   ob=w.get_observations();standing=actor.diagnostic_components(ob)["standing_base_action"];b=0 if step<settle else mj((step-settle+1)/ramp);_,_,done,_=w.step(standing+off*b);active&=~done.bool()
   if step>=settle+ramp:
    force=contact.data.net_forces_w_history.torch[:,:,sids,:].norm(dim=-1).amax(dim=1);force_sum+=force;contact_sum+=(force>5);tilt_max=torch.maximum(tilt_max,torch.linalg.vector_norm(robot.data.projected_gravity_b.torch[:,:2],dim=1))
  avg=force_sum/hold;duty=contact_sum/hold
  for k,row in enumerate(rows):
   l=int(lead[k]);s=1-l;row.update({"lead_force_n":float(avg[k,l]),"support_force_n":float(avg[k,s]),"lead_contact_duty":float(duty[k,l]),"support_contact_duty":float(duty[k,s]),"tilt_max_rad":float(tilt_max[k]),"fall":not bool(active[k])});row["viable_weight_shift"]=bool(active[k] and duty[k,s]>=.95 and avg[k,l]/avg[k,s].clamp_min(1e-6)<=.20 and tilt_max[k]<.20)
  fields=list(rows[0]);f=open(out/"weight_shift.csv","w",newline="",encoding="utf-8");dw=csv.DictWriter(f,fieldnames=fields);dw.writeheader();dw.writerows(rows);f.close();v=[x for x in rows if x["viable_weight_shift"]];v.sort(key=lambda x:x["lead_force_n"]/max(x["support_force_n"],1e-6));(out/"summary.json").write_text(json.dumps({"candidate_count":n,"viable_count":len(v),"best":v[:20]},indent=2)+"\n");print(json.dumps({"viable_count":len(v),"best":v[:3]},indent=2));raw.close()
if __name__=="__main__":main()
