"""ReplayV2 evaluator assigning up to three 300-episode conditions to one 1024-env batch."""
from __future__ import annotations
import argparse,csv,json,math,sys
from pathlib import Path
import gymnasium as gym
import torch

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]; EXP=HERE.parent.parent
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
M0=BASE/"phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight"
STOP=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
sys.path[:0]=[str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),str(EXP/"src"),str(HERE.parent)]
import isaaclab_tasks  # noqa:F401,E402
import g1_omnidirectional.tasks  # noqa:F401,E402
from g1_omnidirectional.policy import FrozenGaitActor  # noqa:E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa:E402
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli  # noqa:E402
from w2_p1_a5_common import reproduce_a4  # noqa:E402

p=argparse.ArgumentParser(); p.add_argument("--policy",required=True);p.add_argument("--batch",type=int,required=True);p.add_argument("--split",choices=("validation","heldout"),required=True);p.add_argument("--conditions",help="comma-separated matrix indices");p.add_argument("--condition-specs-json");p.add_argument("--episodes",type=int,default=300);p.add_argument("--output-dir",required=True);p.add_argument("--candidate-takeover-horizon",type=int);add_launcher_args(p);args,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0],*hydra]
if args.condition_specs_json:
 specs=json.loads(Path(args.condition_specs_json).read_text())
else:
 conditions=[int(x) for x in args.conditions.split(",")]
 specs=[{"id":f"{c:02d}","direction":float((c//3)*45),"speed":.3,"yaw":float((-.3,0.,.3)[c%3])} for c in conditions]
assert specs and len(specs)*args.episodes<=939
N=1024;RAMP=75;HOLD=200;DT=.02
def mj(x): x=x.clamp(0,1);return 10*x**3-15*x**4+6*x**5
def sustained(trace,width):
 s=torch.zeros(trace.shape[1],dtype=torch.long,device=trace.device);first=torch.full_like(s,-1)
 for t in range(trace.shape[0]):
  s=torch.where(trace[t],s+1,torch.zeros_like(s));hit=(s>=width)&(first<0);first[hit]=t-width+1
 return first,first>=0
cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=N;cfg.episode_length_s=12.;cfg.seed=20278501;cfg.observations.policy.enable_corruption=False
if args.device:cfg.sim.device=agent.device=args.device
masks=json.loads((M0/"a7_environment_masks.json").read_text())["batches"][str(args.batch)];available=torch.nonzero(torch.tensor(masks[f"{args.split}_mask"],dtype=torch.bool)).flatten();assert len(available)>=len(specs)*args.episodes
groups={i:available[i*args.episodes:(i+1)*args.episodes] for i in range(len(specs))};active_cpu=torch.zeros(N,dtype=torch.bool)
for ids in groups.values():active_cpu[ids]=True
out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
with launch_simulation(cfg,args):
 wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);env=wrapped.unwrapped;robot=env.scene["robot"];sensor=env.scene["contact_forces"];sf=sensor.find_bodies(".*_ankle_roll_link")[0];rf=robot.find_bodies(".*_ankle_roll_link")[0];cmd=env.command_manager.get_term("base_velocity");cmd.external_override_enabled=True;stop=FrozenGaitActor(STOP).to(env.device).eval();gait=torch.zeros(N,device=env.device);all_ids=torch.arange(N,device=env.device);limits=robot.data.joint_vel_limits;limits=limits[...,1].abs() if limits.ndim==3 else limits
 for _ in range(args.batch+1):
  env.reset(env_ids=all_ids);cmd.external_override.zero_();cmd._update_command();obs=wrapped.get_observations().to(env.device)
  for _ in range(150):
   with torch.inference_mode():a=stop(obs["policy"],gait)
   obs,_,_,_=wrapped.step(a);obs=obs.to(env.device)
 policy=FrozenGaitActor(Path(args.policy)).to(env.device).eval();active=active_cpu.to(env.device);target=torch.zeros(N,3,device=env.device);candidate=None
 if args.candidate_takeover_horizon is not None:
  candidate,fingerprint,_,_,_=reproduce_a4(torch.device(env.device));candidate=candidate.to(env.device).eval()
  if fingerprint["tensor_hash"]!="db65a3069d665b8012fd9d264b7fd54e629a22d25b05a9ff793e23bfc549ac5f":raise RuntimeError("A4 reproduction mismatch")
 for c,ids_cpu in groups.items():
  spec=specs[c];d=spec["direction"];y=spec["yaw"];speed=spec.get("speed",.3);rad=math.radians(d);ids=ids_cpu.to(env.device);target[ids,0]=speed*math.cos(rad);target[ids,1]=speed*math.sin(rad);target[ids,2]=y
 if candidate is not None:
  cmd.external_override.zero_();cmd._update_command();obs=wrapped.get_observations().to(env.device)
  with torch.inference_mode():ca=candidate(obs["policy"],gait);ha=stop(obs["policy"],gait);a=torch.where(active[:,None],ca,ha)
  obs,_,_,_=wrapped.step(a);obs=obs.to(env.device)
 fall=torch.zeros(N,dtype=torch.bool,device=env.device);slip=fall.clone();impact=fall.clone();sat=fall.clone();ss=torch.zeros(N,dtype=torch.long,device=env.device);sats=ss.clone();ec=torch.zeros(N,dtype=torch.long,device=env.device);ev=torch.zeros(N,2,device=env.device);eve=torch.zeros(N,device=env.device);ey=torch.zeros(N,device=env.device);eye=torch.zeros(N,device=env.device);tr={k:[] for k in ("translation","direction","yaw","gait_safety","combined")}
 for step in range(RAMP+HOLD):
  physical=target*mj(torch.tensor(step/RAMP,device=env.device));actor=physical.clone();actor[:,2]=torch.where(actor[:,2]>0,actor[:,2]*1.5,actor[:,2]);cmd.external_override.zero_();cmd.external_override[active]=actor[active];cmd._update_command();obs=wrapped.get_observations().to(env.device)
  with torch.inference_mode():
   pa=policy(obs["policy"],gait);ha=stop(obs["policy"],gait)
   if candidate is not None:
    ca=candidate(obs["policy"],gait);pa=pa if step<args.candidate_takeover_horizon else ca
   a=torch.where(active[:,None],pa,ha)
  obs,_,done,extras=wrapped.step(a);obs=obs.to(env.device);timeout=extras.get("time_outs",torch.zeros_like(done)).bool();fall|=done.bool()&~timeout;force=sensor.data.net_forces_w_history[:,-1,sf,:].norm(dim=-1);contact=force>5;fs=torch.linalg.vector_norm(robot.data.body_lin_vel_w[:,rf,:2],dim=-1);bad=((fs>.55)&contact).any(1);ss=torch.where(bad,ss+1,torch.zeros_like(ss));slip|=ss>=5;impact|=force.amax(1)>3500;vr=robot.data.joint_vel.abs().div(limits.clamp_min(1e-6)).amax(1);sats=torch.where(vr>.95,sats+1,torch.zeros_like(sats));sat|=sats>=5
  actual=robot.data.root_lin_vel_b[:,:2];ay=robot.data.root_ang_vel_b[:,2];ve=torch.linalg.vector_norm(actual-physical[:,:2],dim=1);asp=torch.linalg.vector_norm(actual,dim=1);tsp=torch.linalg.vector_norm(physical[:,:2],dim=1);ta=torch.atan2(physical[:,1],physical[:,0]);aa=torch.atan2(actual[:,1],actual[:,0]);de=torch.atan2(torch.sin(aa-ta),torch.cos(aa-ta)).abs()*180/math.pi;tp=torch.where(tsp<1e-8,asp<=.08,ve<=.25);dp=torch.where(tsp<1e-8,torch.ones_like(tp),de<=25);yp=torch.where(physical[:,2].abs()<1e-8,ay.abs()<=.2,(torch.sign(ay)==torch.sign(physical[:,2]))&((ay-physical[:,2]).abs()<=.2));gp=contact.any(1)&~fall&~slip&~impact;combined=tp&dp&yp&gp
  if step>=RAMP:
   for k,v in (("translation",tp),("direction",dp),("yaw",yp),("gait_safety",gp),("combined",combined)):tr[k].append(v.clone())
  if step>=175:ec+=active;ev+=actual*active[:,None];eve+=ve*active;ey+=ay*active;eye+=(ay-target[:,2]).abs()*active
 traces={k:torch.stack(v) for k,v in tr.items()};ec=ec.clamp_min(1);mv=ev/ec[:,None];my=ey/ec;me=eve/ec;mye=eye/ec;ms=torch.linalg.vector_norm(mv,dim=1);ma=torch.atan2(mv[:,1],mv[:,0]);tta=torch.atan2(target[:,1],target[:,0]);mde=torch.atan2(torch.sin(ma-tta),torch.cos(ma-tta)).abs()*180/math.pi;ts=torch.linalg.vector_norm(target[:,:2],dim=1);et=torch.where(ts<1e-8,ms<=.08,(me<=.25)&(mde<=25));eyo=torch.where(target[:,2].abs()<1e-8,my.abs()<=.2,(torch.sign(my)==torch.sign(target[:,2]))&(mye<=.2));eok=et&eyo&~fall&~slip&~impact&~sat
 for c,ids_cpu in groups.items():
  spec=specs[c]
  ids=ids_cpu.to(env.device);cy=traces["yaw"][:150,ids];longest=torch.zeros(len(ids),dtype=torch.long,device=env.device);streak=longest.clone();resets=longest.clone()
  for f in cy:resets+=(~f)&(streak>0);streak=torch.where(f,streak+1,torch.zeros_like(streak));longest=torch.maximum(longest,streak)
  row={"group":"start_matrix","direction":float(spec["direction"]),"speed":float(spec.get("speed",.3)),"yaw":float(spec["yaw"]),"episodes":len(ids),"endpoint_success":float(eok[ids].float().mean()),"yaw_timer_resets":float(resets.float().mean()),"longest_yaw_pass_s":float(longest.float().mean()*DT),"yaw_mae":float(mye[ids].mean()),"fall_rate":float(fall[ids].float().mean()),"dangerous_slip_rate":float(slip[ids].float().mean()),"impact_rate":float(impact[ids].float().mean()),"saturation_rate":float(sat[ids].float().mean())}
  for width,label in ((5,"0p10"),(8,"0p15"),(10,"0p20"),(13,"0p25")):
   first,passed=sustained(traces["combined"][:150,ids],width);valid=first[passed].float()*DT;row[f"acquisition_{label}"]=float(passed.float().mean());row[f"acquisition_{label}_median_s"]=float(valid.median()) if len(valid) else None;row[f"acquisition_{label}_p95_s"]=float(torch.quantile(valid,.95)) if len(valid) else None
  for k,t in traces.items():
   _,passed=sustained(t[:150,ids],10);row[f"{k}_sustained_0p20"]=float(passed.float().mean());row[f"{k}_final_hold_pass_fraction"]=float(t[-100:,ids].float().mean())
  csvpath=out/f"condition_{spec['id']}.csv"
  with csvpath.open("w",newline="",encoding="utf-8") as s:w=csv.DictWriter(s,fieldnames=list(row));w.writeheader();w.writerow(row)
  csvpath.with_suffix(".json").write_text(json.dumps({"policy":args.policy,"batch":args.batch,"split":args.split,"row":row,"multi_condition_specs":specs,"candidate_takeover_horizon":args.candidate_takeover_horizon},indent=2)+"\n")
 print(json.dumps({"condition_specs":specs,"status":"COMPLETE"}),flush=True);wrapped.close()
