"""Build the D7 R4 oracle distillation dataset and seal a new held-out split."""
from __future__ import annotations
import argparse,collections,hashlib,importlib.util,json,math,sys
from pathlib import Path
import gymnasium as gym
import torch
HERE=Path(__file__).resolve();REPO=HERE.parents[4];D6=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d6_omnidirectional_stop_teacher/raw";OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7_r4_stop_oracle_distillation";RAW=OUT/"raw";DATA=RAW/"dataset"
def load(name,path):s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d6=load("d6",HERE.parent/"run_phase2_d6_audit.py");pre=load("d7pre",HERE.parent/"run_phase2_d7_preflight.py");d3=d6.d3
from g1_omnidirectional.policy import FrozenGaitActor  # noqa:E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa:E402
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli  # noqa:E402
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def fsha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def episode_key(split,cond,recipe):return f"{split}:c{int(cond):02d}:r{int(recipe):03d}"
def collect_pre(world,walk,paths):
 out=[];gait=torch.zeros(world.env.num_envs,device=world.device)
 for p in paths:
  x=torch.load(p,map_location="cpu",weights_only=False);d6.restore_payload(world,x);base=world.env.observation_manager.compute()["policy"];obs=world.obs()
  with torch.inference_mode():a=walk(base,gait)
  for j,e in enumerate(x["entries"]):
   if not x["w_move_acquired"][j]:continue
   out.append({"key":episode_key(e["split"],e["condition"]["condition_id"],e["recipe_id"]),"observation_141":obs[j].cpu(),"action_37":a[j].cpu(),"condition_id":e["condition"]["condition_id"],"recipe_id":e["recipe_id"],"split":e["split"],"physical_command":world.state.physical_command[j].cpu(),"previous_command":world.state.previous_physical_command[j].cpu(),"previous_action":world.env.action_manager.prev_action[j].cpu()})
 return out
def held_conditions():
 base=d6.conditions();out=[]
 for c in base:
  count=18 if c["condition_id"]==0 else 17
  for i in range(count):
   pure=c["kind"]=="pure_yaw";k=i%9;rep=i//9;dd=(-5,0,5)[k%3];ds=(-.05,0,.05)[(k//3)%3];dy=(-.04,-.03,-.02,-.01,0,.01,.02,.03,.04)[k];sw=(.45,.50,.55)[k%3]
   out.append({**c,"formal_condition_id":c["condition_id"],"variant":i,"condition_id":c["condition_id"],"direction_deg":c["direction_deg"]+(0 if pure else dd),"speed":0 if pure else c["speed"]+ds,"yaw":c["yaw"]+dy,"switch_time_s":sw,"switch_step":math.ceil(sw/.02)})
 return out
def context_for(step,checkpoint,last_stop):
 if step==0:return 1
 if checkpoint==0:return 3 if step>=21 else 2
 if checkpoint==2:return 6
 if step>=last_stop-24:return 5
 return 3 if step<=29 else 4
def build_dataset(pre_rows):
 premap={r["key"]:r for r in pre_rows};segments=[]
 for p in sorted((D6/"labels/selected").glob("*.pt")):
  x=torch.load(p,map_location="cpu",weights_only=False);starts=torch.where(x["control_step"]==0)[0].tolist();starts.append(len(x["control_step"]))
  for a,b in zip(starts[:-1],starts[1:]):
   split=int(x["split"][a]);
   if split==2:continue
   cond=int(x["condition_id"][a]);recipe=int(x["recipe_id"][a]);key=episode_key("train" if split==0 else "validation",cond,recipe);segments.append((key,split,cond,recipe,{k:v[a:b] for k,v in x.items()}))
 train=[s for s in segments if s[1]==0];valall=[s for s in segments if s[1]==1];by=collections.defaultdict(list)
 for s in valall:by[s[2]].append(s)
 val=[]
 for c in range(34):val.extend(sorted(by[c],key=lambda s:s[3])[:18 if c==0 else 17])
 assert len(train)==2702 and len(val)==579,(len(train),len(val))
 def pack(items,split_name):
  fields={k:[] for k in ("observation_141","action_37","recipe_id","condition_id","control_step","time_since_stop_request","target_mode","previous_mode","current_command","previous_command","command_delta","ramp_progress","previous_action","teacher_checkpoint_id","context_id","episode_index")};episodes=[]
  for ei,(key,_,cond,recipe,x) in enumerate(items):
   if key in premap:
    p=premap[key];fields["observation_141"].append(p["observation_141"][None]);fields["action_37"].append(p["action_37"][None]);fields["recipe_id"].append(torch.tensor([recipe]));fields["condition_id"].append(torch.tensor([cond]));fields["control_step"].append(torch.tensor([-1]));fields["time_since_stop_request"].append(torch.tensor([-.02]));fields["target_mode"].append(torch.tensor([1]));fields["previous_mode"].append(torch.tensor([1]));fields["current_command"].append(p["physical_command"][None]);fields["previous_command"].append(p["previous_command"][None]);fields["command_delta"].append(torch.zeros(1,3));fields["ramp_progress"].append(torch.ones(1,1));fields["previous_action"].append(p["previous_action"][None]);fields["teacher_checkpoint_id"].append(torch.tensor([0]));fields["context_id"].append(torch.tensor([0]));fields["episode_index"].append(torch.tensor([ei]))
   steps=x["control_step"];cp=x["checkpoint_id"];last_stop=int(steps[cp==1].max()) if bool((cp==1).any()) else 0;n=len(steps);fields["observation_141"].append(x["observation_141"]);fields["action_37"].append(x["action_37"]);fields["recipe_id"].append(x["recipe_id"]);fields["condition_id"].append(x["condition_id"]);fields["control_step"].append(steps);fields["time_since_stop_request"].append(steps.float()*.02);fields["target_mode"].append(x["motion_mode"]);fields["previous_mode"].append(x["previous_mode"]);fields["current_command"].append(x["physical_command"]);fields["previous_command"].append(x["previous_command"]);fields["command_delta"].append(x["command_delta"]);fields["ramp_progress"].append(x["ramp_progress"]);fields["previous_action"].append(x["previous_action"]);fields["teacher_checkpoint_id"].append(cp);fields["context_id"].append(torch.tensor([context_for(int(st),int(ci),last_stop) for st,ci in zip(steps,cp)]));fields["episode_index"].append(torch.full((n,),ei));episodes.append({"episode_index":ei,"episode_id":key,"recipe_id":recipe,"condition_id":cond,"samples":n+(1 if key in premap else 0)})
  packed={k:torch.cat(v) for k,v in fields.items()};packed["split_name"]=split_name;packed["episodes"]=episodes;return packed
 trainp=pack(train,"train");valp=pack(val,"validation");DATA.mkdir(parents=True,exist_ok=True);torch.save(trainp,DATA/"train.pt");torch.save(valp,DATA/"validation.pt");return trainp,valp
def audit_dataset(train,val):
 conflict={};coef=torch.randint(-(2**31),2**31-1,(141,),generator=torch.Generator().manual_seed(20279102),dtype=torch.int64)
 obs=torch.cat((train["observation_141"],val["observation_141"]));act=torch.cat((train["action_37"],val["action_37"]))
 for q in (1e-6,1e-5,1e-4,1e-3):
  qo=torch.round(obs/q).to(torch.int64);h=(qo*coef).sum(1);order=torch.argsort(h);hs=h[order];material=collisions=0
  for pos in torch.where(hs[1:]==hs[:-1])[0].tolist():
   i=int(order[pos]);j=int(order[pos+1]);
   if not torch.equal(qo[i],qo[j]):continue
   collisions+=1;l2=float((act[i]-act[j]).norm());cos=float(torch.nn.functional.cosine_similarity(act[i:i+1],act[j:j+1]));material+=int(l2>=.5 or cos<=.98)
  conflict[str(q)]={"collisions":collisions,"material_conflicts":material}
 # Paired boundary neighbors within each trajectory.
 nn=[]
 for data in (train,val):
  for ei in range(len(data["episodes"])):
   idx=torch.where(data["episode_index"]==ei)[0];cp=data["teacher_checkpoint_id"][idx];change=torch.where(cp[1:]!=cp[:-1])[0]
   for z in change.tolist():
    i=int(idx[z]);j=int(idx[z+1]);d=data["observation_141"][j]-data["observation_141"][i];a=data["action_37"];nn.append({"episode_id":data["episodes"][ei]["episode_id"],"teacher_role_pair":f"{int(data['teacher_checkpoint_id'][i])}->{int(data['teacher_checkpoint_id'][j])}","input_distance_141":float(d.norm()),"physical_state_only_distance":float(d[:124].norm()),"command_history_distance":float(d[124:].norm()),"action_l2":float((a[j]-a[i]).norm()),"action_cosine":float(torch.nn.functional.cosine_similarity(a[j:j+1],a[i:i+1]))})
 return conflict,nn
def main():
 p=argparse.ArgumentParser();add_launcher_args(p);args,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0],*hydra];RAW.mkdir(parents=True,exist_ok=True);cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=476;cfg.seed=20279103;cfg.episode_length_s=20.;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None
 if args.device:cfg.sim.device=agent.device=args.device
 resets=d3.load_resets();sev=torch.zeros(680)
 with launch_simulation(cfg,args):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=d3.StandWorld(wrapped,resets,sev);walk=FrozenGaitActor(d3.WMOVE).to(world.device).eval();train_paths=sorted((D6/"snapshots/selected").glob("train_batch_*.pt"));val_paths=sorted(D6.glob("validation_snapshot_batch_*.pt"));pre_rows=collect_pre(world,walk,train_paths+val_paths);torch.save(pre_rows,RAW/"pre_stop_samples.pt");hc=held_conditions();last=d3.HELDOUT[-2:];entries=[{"episode_id":f"held:c{c['formal_condition_id']:02d}:v{c['variant']:02d}","recipe_id":last[i%2],"split":"sealed-held-out","condition":c} for i,c in enumerate(hc)];payloads=[pre.generate(world,walk,entries[i:i+world.env.num_envs]) for i in range(0,len(entries),world.env.num_envs)];torch.save(payloads,RAW/"sealed_heldout_snapshots.pt");seal={"seed":20279103,"episodes":len(entries),"batches":len(payloads),"conditions":34,"snapshot_sha256":fsha(RAW/"sealed_heldout_snapshots.pt"),"labels_opened":False,"outcomes_opened":False,"created_before_student_training":True};dump(RAW/"heldout_seal.json",seal);wrapped.close()
 train,val=build_dataset(pre_rows);conf,nn=audit_dataset(train,val);summary={"name":"Exp014R4StopOracleDistillationDatasetV1","episodes":{"train":len(train["episodes"]),"validation":len(val["episodes"]),"sealed-held-out":len(entries)},"ratio":{"train":len(train["episodes"])/(len(train["episodes"])+len(val["episodes"])+len(entries)),"validation":len(val["episodes"])/(len(train["episodes"])+len(val["episodes"])+len(entries)),"sealed-held-out":len(entries)/(len(train["episodes"])+len(val["episodes"])+len(entries))},"samples":{"train":len(train["observation_141"]),"validation":len(val["observation_141"])},"hashes":{"train":fsha(DATA/"train.pt"),"validation":fsha(DATA/"validation.pt"),"sealed-held-out":seal["snapshot_sha256"]},"conflicts":conf,"nearest_neighbors":nn};dump(RAW/"dataset_results.json",summary)
 print(json.dumps({k:summary[k] for k in ("episodes","ratio","samples","hashes","conflicts")},indent=2))
if __name__=="__main__":main()
