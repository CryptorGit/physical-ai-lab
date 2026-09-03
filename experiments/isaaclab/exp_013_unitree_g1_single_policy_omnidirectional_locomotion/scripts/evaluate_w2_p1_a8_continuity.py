"""B0-B8 counterfactual action continuity at frozen A8 checkpoint boundaries."""
from __future__ import annotations
import csv,json,math,sys
from pathlib import Path
import gymnasium as gym
import torch
HERE=Path(__file__).resolve();REPO=HERE.parents[4];EXP=HERE.parent.parent;BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion";OUT=BASE/"phase_w2_p1_a8_offline_start_teacher_oracle";M0=BASE/"phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight";R2=BASE/"phase_w2_p1_a7_r2_rear_yaw_start_teacher_replay_v2";STOP=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
sys.path[:0]=[str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),str(EXP/"src")]
import isaaclab_tasks,g1_omnidirectional.tasks  # noqa:E402,F401
from g1_omnidirectional.policy import FrozenGaitActor  # noqa:E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa:E402
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli  # noqa:E402
import argparse
p=argparse.ArgumentParser();add_launcher_args(p);a,h=setup_preset_cli(p);sys.argv=[sys.argv[0],*h]
mapping=json.loads((OUT/"offline_start_teacher_condition_map_v1.json").read_text())["condition_map"];lookup={(int(x["physical_command"]["direction_deg"]),float(x["physical_command"]["yaw_radps"])):x["selected_checkpoint_update"] for x in mapping};edges=[]
for d in range(0,360,45):
 for y in (-.3,0.,.3):
  b=((d+45)%360,y)
  if lookup[d,y]!=lookup[b]:edges.append(((d,y),b,"direction"))
for d in range(0,360,45):
 for y1,y2 in ((-.3,0.),(0.,.3)):
  if lookup[d,y1]!=lookup[d,y2]:edges.append(((d,y1),(d,y2),"yaw"))
N=1024;PER=50;cfg,ac=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=N;cfg.episode_length_s=8.;cfg.seed=20278501;cfg.observations.policy.enable_corruption=False
if a.device:cfg.sim.device=ac.device=a.device
masks=json.loads((M0/"a7_environment_masks.json").read_text())["batches"]["4"];available=torch.nonzero(torch.tensor(masks["validation_mask"],dtype=torch.bool)).flatten();groups={i:available[i*PER:(i+1)*PER] for i in range(len(edges))}
with launch_simulation(cfg,a):
 w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=ac.clip_actions);env=w.unwrapped;term=env.command_manager.get_term("base_velocity");term.external_override_enabled=True;g=torch.zeros(N,device=env.device);ids=torch.arange(N,device=env.device);stop=FrozenGaitActor(STOP).to(env.device).eval()
 for _ in range(5):
  env.reset(env_ids=ids);term.external_override.zero_();term._update_command();obs=w.get_observations().to(env.device)
  for _ in range(150):
   with torch.inference_mode():act=stop(obs["policy"],g)
   obs,_,_,_=w.step(act);obs=obs.to(env.device)
 p10=FrozenGaitActor(R2/"checkpoints/model_010.pt").to(env.device).eval();p150=FrozenGaitActor(R2/"checkpoints/model_150.pt").to(env.device).eval();active=torch.zeros(N,dtype=torch.bool,device=env.device);target=torch.zeros(N,3,device=env.device)
 for i,(left,_,_) in enumerate(edges):
  q=groups[i].to(env.device);active[q]=True;rad=math.radians(left[0]);target[q,0]=.3*math.cos(rad);target[q,1]=.3*math.sin(rad);target[q,2]=left[1]
 rows=[];previous={}
 for step in range(9):
  alpha=torch.tensor(step/75,device=env.device).clamp(0,1);alpha=10*alpha**3-15*alpha**4+6*alpha**5;physical=target*alpha;actor=physical.clone();actor[:,2]=torch.where(actor[:,2]>0,1.5*actor[:,2],actor[:,2]);term.external_override.zero_();term.external_override[active]=actor[active];term._update_command();obs=w.get_observations().to(env.device)
  with torch.inference_mode():a10=p10(obs["policy"],g);a150=p150(obs["policy"],g);house=stop(obs["policy"],g)
  for i,(left,right,kind) in enumerate(edges):
   q=groups[i].to(env.device);x=a10[q];y=a150[q];diff=x-y;l2=torch.linalg.vector_norm(diff,dim=1);cos=torch.nn.functional.cosine_similarity(x,y,dim=1);prev10=previous.get((i,10),x);prev150=previous.get((i,150),y);d10=torch.linalg.vector_norm(x-prev10,dim=1);d150=torch.linalg.vector_norm(y-prev150,dim=1);previous[i,10]=x.clone();previous[i,150]=y.clone();contrib={"legs":diff[:,:18].square().sum(1).sqrt(),"waist":diff[:,18:21].square().sum(1).sqrt(),"torso_arms":diff[:,21:33].square().sum(1).sqrt(),"hands":diff[:,33:37].square().sum(1).sqrt()};rows.append({"boundary_id":i,"boundary_kind":kind,"left_direction":left[0],"left_yaw":left[1],"left_update":lookup[left],"right_direction":right[0],"right_yaw":right[1],"right_update":lookup[right],"boundary_step":step,"whole_body_l2":float(l2.mean()),"action_cosine":float(cos.mean()),"update10_action_derivative_l2":float(d10.mean()),"update150_action_derivative_l2":float(d150.mean()),"pd_target_derivative_proxy":"normalized action derivative; actuator scale unchanged",**{f"{k}_contribution":float(v.mean()) for k,v in contrib.items()},"continuity_warning":bool(float(cos.mean())<.85 or float(l2.mean())>4.)})
  action=house.clone()
  for i,(left,_,_) in enumerate(edges):
   q=groups[i].to(env.device);action[q]=a10[q] if lookup[left]==10 else a150[q]
  obs,_,_,_=w.step(action);obs=obs.to(env.device)
 with (OUT/"offline_teacher_condition_boundary_continuity.csv").open("w",newline="",encoding="utf-8") as s:wr=csv.DictWriter(s,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 result={"status":"WARNING" if any(r["continuity_warning"] for r in rows) else "PASS","checkpoint_boundary_count":len(edges),"warning_count":sum(r["continuity_warning"] for r in rows),"maximum_action_l2":max(r["whole_body_l2"] for r in rows),"minimum_action_cosine":min(r["action_cosine"] for r in rows),"joint_groups":{"legs":"0:18","waist":"18:21","torso_arms":"21:33","hands":"33:37"},"rows":rows};(OUT/"offline_teacher_condition_boundary_continuity.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({k:result[k] for k in ("status","warning_count","maximum_action_l2","minimum_action_cosine")}));w.close()
