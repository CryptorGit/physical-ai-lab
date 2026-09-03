"""D6 frozen R4 validation, held-out, parity, and label authorization runtime."""
from __future__ import annotations
import argparse,collections,hashlib,importlib.util,json,sys
from pathlib import Path
import gymnasium as gym
import torch

HERE=Path(__file__).resolve();REPO=HERE.parents[4];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d6_omnidirectional_stop_teacher";RAW=OUT/"raw"
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
audit=load_module("d6audit",HERE.parent/"run_phase2_d6_audit.py");d3=audit.d3
from g1_omnidirectional.policy import FrozenGaitActor  # noqa:E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa:E402
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli  # noqa:E402

SHA={0:"61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d",1:"66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698",2:"734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"}
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def jhash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def make_entries(split):
 conds=audit.conditions();out=[]
 for c in conds:
  if split=="train":pool=d3.TRAIN;start=(c["condition_id"]*80)%len(pool);recipes=(pool+pool)[start:start+80]
  elif split=="validation":recipes=d3.VALIDATION[:100]
  else:recipes=d3.HELDOUT[:100]
  for i,r in enumerate(recipes):out.append({"snapshot_id":f"{split}_c{c['condition_id']:02d}_e{i:03d}","split":split,"recipe_id":r,"condition":c})
 return out
def generate(world,walk,split,tag="selected"):
 entries=make_entries(split);payloads=[];manifest=[]
 for b,start in enumerate(range(0,len(entries),world.env.num_envs)):
  path=RAW/"snapshots"/tag/f"{split}_batch_{b:02d}.pt";payload,p=audit.generate_batch(world,walk,entries[start:start+world.env.num_envs],b,path);payloads.append(payload);manifest.append({"batch":b,"path":str(p.relative_to(REPO)).replace("\\","/"),"active":payload["active"],"snapshot_hash":payload["snapshot_hash"],"w_move_acquired":sum(payload["w_move_acquired"])})
 return payloads,manifest
def load_validation():return [torch.load(p,map_location="cpu",weights_only=False) for p in sorted(RAW.glob("validation_snapshot_batch_*.pt"))]
def run_route(world,payloads,walk,stop,hold,split,capture=False,tag="selected"):
 rows=[];labels=[]
 for b,payload in enumerate(payloads):
  lp=RAW/"labels"/tag/f"{split}_batch_{b:02d}.pt" if capture else None;rows.extend(audit.route_batch(world,payload,"R4_W_MOVE_STAGE2Q_HOLD",walk,stop,hold,lp));labels.append(lp) if lp else None
 return audit.summarize(rows,"R4_W_MOVE_STAGE2Q_HOLD"),rows,labels
def parity_payload(summary,rows,manifest):
 classes=[(r["stop_acquisition"],r["stand_after_stop"],r["joint_success"]) for r in rows if r["w_move_start_acquired"]];acq=[r["acquisition_time_s"] for r in rows if r["w_move_start_acquired"]]
 return {"moving_snapshot_hashes":[m["snapshot_hash"] for m in manifest],"moving_snapshot_hashes_hash":jhash([m["snapshot_hash"] for m in manifest]),"action_hashes":[r["action_hash"] for r in rows],"action_hashes_hash":jhash([r["action_hash"] for r in rows]),"acquisition_classifications_hash":jhash(classes),"acquisition_times_hash":jhash(acq),"handoff_classifications_hash":jhash([r["stand_after_stop"] for r in rows]),"aggregate_metrics":summary,"aggregate_metrics_hash":jhash(summary)}
def label_audit(paths):
 samples=missing=nan=bad=0;per_split=collections.Counter();per_role=collections.Counter();files=[]
 for p in paths:
  x=torch.load(p,map_location="cpu",weights_only=False);n=len(x["recipe_id"]);samples+=n;nan+=sum(int(not torch.isfinite(v).all()) for v in x.values() if torch.is_tensor(v) and v.is_floating_point());bad+=int((x["action_37"].abs()>100).any());per_split.update(x["split"].tolist());per_role.update(x["teacher_role_id"].tolist());files.append({"path":str(p.relative_to(REPO)).replace("\\","/"),"sha256":file_sha(p),"samples":n})
 conflict={};g=torch.Generator().manual_seed(20279001);coef=torch.randint(-(2**31),2**31-1,(141,),generator=g,dtype=torch.int64)
 for q in (1e-6,1e-5,1e-4,1e-3):
  counts=collections.Counter()
  for p in paths:
   x=torch.load(p,map_location="cpu",weights_only=False);h=(torch.round(x["observation_141"]/q).to(torch.int64)*coef).sum(1);counts.update(h.tolist())
  dup={h for h,c in counts.items() if c>1};reps={};material=collisions=0
  if dup:
   for p in paths:
    x=torch.load(p,map_location="cpu",weights_only=False);qo=torch.round(x["observation_141"]/q).to(torch.int32);hs=(qo.to(torch.int64)*coef).sum(1)
    for i,h in enumerate(hs.tolist()):
     if h not in dup:continue
     key=(h,qo[i].numpy().tobytes());a=x["action_37"][i]
     if key in reps:
      collisions+=1;b=reps[key];l2=float((a-b).norm());cos=float(torch.nn.functional.cosine_similarity(a[None],b[None]))
      if l2>=.5 or cos<=.98:material+=1
     else:reps[key]=a
  conflict[str(q)]={"quantized_input_collisions":collisions,"material_conflicts":material}
 return {"samples":samples,"missing":missing,"nan_inf":nan,"bounds_violation":bad,"split_samples":{"train":per_split[0],"validation":per_split[1],"held-out":per_split[2]},"role_samples":{"S_STOP_OMNI":per_role[0],"S_HOLD":per_role[1]},"files":files,"quantization":conflict,"material_conflicts":sum(x["material_conflicts"] for x in conflict.values()),"teacher_role_map":{"0":"S_STOP_OMNI/R4_W_MOVE_STAGE2Q_HOLD","1":"S_HOLD"},"checkpoint_map":SHA}

def simultaneous_parity(world,walk,stop,hold):
 left=[];right=[];initial_left=[];initial_right=[]
 for c in audit.conditions():
  recipes=d3.VALIDATION[:100];entries=[]
  for cohort in (0,1):
   for i,r in enumerate(recipes):entries.append({"snapshot_id":f"same{cohort}_c{c['condition_id']:02d}_e{i:03d}","split":"validation","recipe_id":r,"condition":c})
  payload,_=audit.generate_batch(world,walk,entries,c["condition_id"],RAW/"snapshots/same_process"/f"condition_{c['condition_id']:02d}.pt");rows=audit.route_batch(world,payload,"R4_W_MOVE_STAGE2Q_HOLD",walk,stop,hold)
  left.extend(rows[:100]);right.extend(rows[100:]);snap=payload["snapshot"]
  for i in range(100):
   initial_left.append(audit.sha_bytes(*(v[i:i+1] for v in snap.values())));initial_right.append(audit.sha_bytes(*(v[i+100:i+101] for v in snap.values())))
 a=audit.summarize(left,"R4_W_MOVE_STAGE2Q_HOLD");b=audit.summarize(right,"R4_W_MOVE_STAGE2Q_HOLD")
 comparison={"recipe_order_equal":[r["recipe_id"] for r in left]==[r["recipe_id"] for r in right],"moving_snapshot_hashes_equal":initial_left==initial_right,"action_hashes_equal":[r["action_hash"] for r in left]==[r["action_hash"] for r in right],"acquisition_classifications_equal":[r["stop_acquisition"] for r in left]==[r["stop_acquisition"] for r in right],"handoff_classifications_equal":[r["stand_after_stop"] for r in left]==[r["stand_after_stop"] for r in right],"aggregate_metrics_equal":a==b,"metric_difference":0 if a==b else None}
 return {"method":"two simultaneous replicated 100-episode cohorts per condition in one scene/process","run_1":a,"run_2":b,"hashes":{"initial_1":jhash(initial_left),"initial_2":jhash(initial_right),"action_1":jhash([r["action_hash"] for r in left]),"action_2":jhash([r["action_hash"] for r in right])},"comparison":comparison,"pass":all(v is True or v==0 for v in comparison.values())}

def main():
 p=argparse.ArgumentParser();p.add_argument("--mode",choices=("main","fresh-parity","same-parity","same-parity-scenes"),default="main");p.add_argument("--run-id",default="fresh_1");add_launcher_args(p);args,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0],*hydra];RAW.mkdir(parents=True,exist_ok=True)
 cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=476;cfg.seed=20279001;cfg.episode_length_s=20.;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None
 if args.device:cfg.sim.device=agent.device=args.device
 resets=d3.load_resets();severity=torch.zeros(680)
 with launch_simulation(cfg,args):
  wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=d3.StandWorld(wrapped,resets,severity);walk=FrozenGaitActor(d3.WMOVE).to(world.device).eval();stop=FrozenGaitActor(d3.P1).to(world.device).eval();hold=d3.initialize("P0_STAND_PARENT",world.device)[0]
  if args.mode=="same-parity-scenes":
   runs=[]
   for k in range(2):
    payloads,manifest=generate(world,walk,"validation",f"same_scene_{k+1}");summary,rows,_=run_route(world,payloads,walk,stop,hold,"validation",False,f"same_scene_{k+1}");runs.append(parity_payload(summary,rows,manifest));wrapped.close()
    if k==0:
     wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=d3.StandWorld(wrapped,resets,severity);walk=FrozenGaitActor(d3.WMOVE).to(world.device).eval();stop=FrozenGaitActor(d3.P1).to(world.device).eval();hold=d3.initialize("P0_STAND_PARENT",world.device)[0]
   keys=("moving_snapshot_hashes_hash","action_hashes_hash","acquisition_classifications_hash","acquisition_times_hash","handoff_classifications_hash","aggregate_metrics_hash");comparison={k:runs[0][k]==runs[1][k] for k in keys};comparison["metric_difference"]=0 if runs[0]["aggregate_metrics"]==runs[1]["aggregate_metrics"] else None;dump(RAW/"parity_same_process_scenes.json",{"method":"two independently constructed scenes in one OS process","runs":runs,"comparison":comparison,"pass":all(v is True or v==0 for v in comparison.values())});return
  if args.mode=="same-parity":dump(RAW/"parity_same_process.json",simultaneous_parity(world,walk,stop,hold));wrapped.close();return
  if args.mode=="fresh-parity":
   payloads,manifest=generate(world,walk,"validation",args.run_id);summary,rows,_=run_route(world,payloads,walk,stop,hold,"validation",False,args.run_id);dump(RAW/f"parity_{args.run_id}.json",parity_payload(summary,rows,manifest));wrapped.close();return
  val_payloads=load_validation();audit_manifest=json.loads((RAW/"audit_results.json").read_text())["snapshot_manifest"]
  held_payloads,held_manifest=generate(world,walk,"held-out");held,held_rows,held_labels=run_route(world,held_payloads,walk,stop,hold,"held-out",True)
  held_pass=held["stop_acquisition"]>=.95 and held["conditional_stand_after_stop"]>=.95 and held["joint_success"]>=.90 and held["fall"]<=.02 and held["dangerous_slip"]<=.05 and held["minimum_condition_joint_success"]>=.80
  result={"selected_route":"R4_W_MOVE_STAGE2Q_HOLD","route_frozen_on_validation":True,"heldout_opened_once":True,"heldout":held,"heldout_rows":held_rows,"heldout_pass":held_pass,"heldout_manifest":held_manifest,"fallback":False}
  if held_pass:
   val,val_rows,val_labels=run_route(world,val_payloads,walk,stop,hold,"validation",True);train_payloads,train_manifest=generate(world,walk,"train");train,train_rows,train_labels=run_route(world,train_payloads,walk,stop,hold,"train",True);result["validation_replay"]=val;result["train_diagnostic"]=train;result["train_manifest"]=train_manifest
   same=[]
   for _ in range(2):s,r,_=run_route(world,val_payloads,walk,stop,hold,"validation");same.append(parity_payload(s,r,audit_manifest))
   result["same_process_parity"]=same;all_labels=train_labels+val_labels+held_labels;result["label_audit"]=label_audit(all_labels);result["label_files"]=[str(x.relative_to(REPO)).replace("\\","/") for x in all_labels]
  dump(RAW/"selected_results.json",result);wrapped.close()
 print(json.dumps({"heldout_pass":result["heldout_pass"],"heldout_joint":held["joint_success"],"labels":result.get("label_audit",{}).get("samples")},indent=2))
if __name__=="__main__":main()
