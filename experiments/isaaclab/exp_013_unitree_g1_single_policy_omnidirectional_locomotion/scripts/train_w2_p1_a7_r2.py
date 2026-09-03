"""Continue the single authorized A7-R2 masked-PPO run after identity-exact update 1."""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,math,subprocess,sys
from collections import OrderedDict
from pathlib import Path
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve();REPO=HERE.parents[4];BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion";OUT=BASE/"phase_w2_p1_a7_r2_rear_yaw_start_teacher_replay_v2";RAW=OUT/"raw";M0=BASE/"phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight";COL=HERE.parent/"collect_w2_p1_a7_r2_window.py";EVAL=HERE.parent/"evaluate_w2_p1_a7_r2.py";ISAAC=Path.home()/"workspace/IsaacLab/isaaclab.bat"
N=1024;T=24;SEED=20278421;OFFSETS=[0,24,48,72,96,120,144,168,192,216,240,251];SAVE={10,20,45,75,100,120,130,140,150};ak=("first_base_weight","first_gait_column","first_bias","hidden.1.weight","hidden.1.bias","hidden.3.weight","hidden.3.bias","hidden.5.weight","hidden.5.bias");ck=("mlp.0.weight","mlp.0.bias","mlp.2.weight","mlp.2.bias","mlp.4.weight","mlp.4.bias","mlp.6.weight","mlp.6.bias")
args=argparse.ArgumentParser();args.add_argument("--resume-update",type=int,default=1);args=args.parse_args()
def hobj(x):
 h=hashlib.sha256()
 def v(z):
  if torch.is_tensor(z):t=z.detach().cpu().contiguous();h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
  elif isinstance(z,dict):
   for k in sorted(z,key=str):h.update(str(k).encode());v(z[k])
  elif isinstance(z,(list,tuple)):
   for q in z:v(q)
  else:h.update(repr(z).encode())
 v(x);return h.hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def phase(u):
 if u<=20:return "R1_REAR_0P15",.15
 if u<=45:return "R2_REAR_0P20",.20
 if u<=75:return "R3_REAR_0P25",.25
 if u<=120:return "R4_REAR_0P30",.30
 return "R5_CONSOLIDATION",None
def largest(count,residual):
 raw=torch.tensor([.6,.2,.2],dtype=torch.float64)*count+torch.tensor(residual);n=torch.floor(raw).long();
 for i in torch.argsort(raw-n,descending=True)[:count-int(n.sum())]:n[i]+=1
 return n.tolist(),(raw-n).tolist()
def targets(train_mask,rear_speed,cursor,residual):
 ids=torch.nonzero(train_mask).flatten();alloc,new=largest(len(ids),residual);rear,other,static=torch.split(ids,alloc);out=torch.zeros(N,3)
 if rear_speed is None:
  counts=[int(len(rear)*x) for x in (.10,.15,.25)];counts.append(len(rear)-sum(counts));start=0
  for speed,n in zip((.15,.20,.25,.30),counts):out[rear[start:start+n],0]=-speed;start+=n
 else:out[rear,0]=-rear_speed
 out[rear,2]=-.3;oc=[]
 for angle in (0,45,90,135,225,270,315):
  r=math.radians(angle)
  for yaw in (-.3,0.,.3):oc.append((.3*math.cos(r),.3*math.sin(r),yaw))
 for j,e in enumerate(other.tolist()):out[e]=torch.tensor(oc[(j+cursor)%len(oc)])
 sc=[]
 for angle in range(0,360,22):r=math.radians(angle);sc.append((.3*math.cos(r),.3*math.sin(r),0.))
 sc += [(0.,0.,-.3),(0.,0.,.3),(.6,0.,0.),(1.2,0.,0.)]
 for angle in range(0,360,45):
  r=math.radians(angle)
  for yaw in (-.3,0.,.3):sc.append((.3*math.cos(r),.3*math.sin(r),yaw))
 for j,e in enumerate(static.tolist()):out[e]=torch.tensor(sc[(j+cursor)%len(sc)])
 mirror=out.clone();mirror[:,1]*=-1;mirror[:,2]*=-1
 return out,mirror,alloc,new
def am(ps,o):
 x=F.elu(F.linear(o[:,:123],ps[0],ps[2])+o[:,123:]@ps[1].T);x=F.elu(F.linear(x,ps[3],ps[4]));x=F.elu(F.linear(x,ps[5],ps[6]));return F.linear(x,ps[7],ps[8])
def cv(ps,o):
 x=F.elu(F.linear(o,ps[0],ps[1]));x=F.elu(F.linear(x,ps[2],ps[3]));x=F.elu(F.linear(x,ps[4],ps[5]));return F.linear(x,ps[6],ps[7]).squeeze(-1)
def compact_pair(paths):
 compact=[];meta=[]
 for path in paths:
  p=torch.load(path,map_location="cpu",weights_only=False);valid=p["valid"].bool();done=p["done"].bool();values=p["old_value"];rewards=p["reward"];adv=torch.zeros_like(values);carry=torch.zeros(N)
  for t in range(T-1,-1,-1):
   nxt=p["last_value"] if t==T-1 else values[t+1];alive=(~done[t]).float();delta=rewards[t]+.99*alive*nxt-values[t];carry=delta+.99*.95*alive*carry;adv[t]=carry
  idx=valid.nonzero();flat=idx[:,0]*N+idx[:,1];ra=adv.flatten()[flat];compact.append({"observation":p["observation"].flatten(0,1)[flat],"action":p["action"].flatten(0,1)[flat],"old_logp":p["old_logp"].flatten()[flat],"old_value":values.flatten()[flat],"advantage":ra,"return":ra+values.flatten()[flat]});meta.append({"valid":len(flat),"inventory_hash":p["inventory_schema_hash_before_policy_load"],"capture_hash":p["capture_schema_hash_before_policy_load"]})
 return {k:torch.cat([x[k] for x in compact]) for k in compact[0]},meta
def ppo_update(params,opt,storage,update,std):
 o=storage["observation"];a=storage["action"];ol=storage["old_logp"];ov=storage["old_value"];ret=storage["return"];ad=storage["advantage"];ad=(ad-ad.mean())/(ad.std()+1e-8);oldmean=am(params[:9],o).detach();gen=torch.Generator().manual_seed(SEED+update);metrics=[]
 for epoch in range(5):
  perm=torch.randperm(len(o),generator=gen)
  for idx in torch.tensor_split(perm,4):
   mean=am(params[:9],o[idx]);value=cv(params[9:],o[idx]);logp=(-.5*(((a[idx]-mean)/std)**2+2*std.log()+math.log(2*math.pi))).sum(-1);ratio=(logp-ol[idx]).exp();sur=torch.maximum(-ad[idx]*ratio,-ad[idx]*ratio.clamp(.8,1.2)).mean();vclip=ov[idx]+(value-ov[idx]).clamp(-.2,.2);vl=torch.maximum((value-ret[idx])**2,(vclip-ret[idx])**2).mean();loss=sur+vl;opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step();metrics.append((float(loss),float(vl),gn))
 with torch.inference_mode():newmean=am(params[:9],o);kl=float((.5*torch.square((newmean-oldmean)/std).sum(-1)).mean());nl=(-.5*(((a-newmean)/std)**2+2*std.log()+math.log(2*math.pi))).sum(-1);ratio=(nl-ol).exp()
 return {"loss":sum(x[0] for x in metrics)/len(metrics),"value_loss":sum(x[1] for x in metrics)/len(metrics),"gradient_norm":max(x[2] for x in metrics),"exact_kl":kl,"clip_fraction":float(((ratio<.8)|(ratio>1.2)).float().mean()),"ratio_p95":float(torch.quantile(ratio,.95)),"ratio_p99":float(torch.quantile(ratio,.99)),"mean_action_shift":float((newmean-oldmean).norm(dim=-1).mean()),"nan_inf":int(not all(math.isfinite(v) for row in metrics for v in row))}
def save_checkpoint(update,params,opt,template,runtime,metrics):
 actor=copy.deepcopy(template["actor_state_dict"]);critic=copy.deepcopy(template["critic_state_dict"])
 for k,p in zip(ak,params[:9]):actor[k]=p.detach().cpu()
 for k,p in zip(ck,params[9:]):critic[k]=p.detach().cpu()
 payload={"iter":update,"actor_state_dict":actor,"critic_state_dict":critic,"optimizer_state_dict":copy.deepcopy(opt.state_dict()),"normalizer_state":copy.deepcopy(template["normalizer_state"]),"sampler_state_dict":copy.deepcopy(template["sampler_state_dict"]),"a7_r2_runtime_state":copy.deepcopy(runtime),"infos":{"phase":phase(update)[0],"learning_rate":1.5e-5,**metrics}};path=OUT/"checkpoints"/f"model_{update:03d}.pt";torch.save(payload,path);return path,payload
def run_collector(policy,targets_path,out,batch,offset,seed):
 cmd=[str(ISAAC),"-p",str(COL),"--policy",str(policy),"--targets",str(targets_path),"--output",str(out),"--batch",str(batch),"--offset",str(offset),"--noise-seed",str(seed),"--headless","--device","cuda:0"]
 with out.with_suffix(".log").open("w",encoding="utf-8") as log:subprocess.run(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
def guard(policy,update):
 out=RAW/f"guard_update_{update:03d}.csv";cmd=[str(ISAAC),"-p",str(EVAL),"--policy",str(policy),"--batch","4","--split","validation","--mode","guard","--output",str(out),"--headless","--device","cuda:0"]
 with out.with_suffix(".log").open("w",encoding="utf-8") as log:subprocess.run(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
 rows=json.loads(out.with_suffix(".json").read_text())["rows"];groups={g:[x for x in rows if x["group"]==g] for g in {x["group"] for x in rows}};z=sum(x["endpoint_success"]>=.9 for x in groups["zero_yaw"]);m=sum(x["endpoint_success"]>=.9 for x in groups["moving_turn"]);f=min(x["endpoint_success"] for x in groups["forward_anchor"]);rear=min(x["endpoint_success"] for x in groups["rear_0p15"]);fall=max(x["fall_rate"] for x in rows);slip=max(x["dangerous_slip_rate"] for x in rows);passed=z>=12 and m>=18 and f>=.85 and rear>=.80 and fall<=.10 and slip<=.30;return {"update":update,"zero_yaw_pass":z,"moving_turn_pass":m,"forward_anchor_min":f,"rear_0p15_min":rear,"max_fall":fall,"max_slip":slip,"pass":passed}
def existing_guard(update):
 out=RAW/f"guard_update_{update:03d}.json";rows=json.loads(out.read_text())["rows"];groups={g:[x for x in rows if x["group"]==g] for g in {x["group"] for x in rows}};z=sum(x["endpoint_success"]>=.9 for x in groups["zero_yaw"]);m=sum(x["endpoint_success"]>=.9 for x in groups["moving_turn"]);f=min(x["endpoint_success"] for x in groups["forward_anchor"]);rear=min(x["endpoint_success"] for x in groups["rear_0p15"]);fall=max(x["fall_rate"] for x in rows);slip=max(x["dangerous_slip_rate"] for x in rows);return {"update":update,"zero_yaw_pass":z,"moving_turn_pass":m,"forward_anchor_min":f,"rear_0p15_min":rear,"max_fall":fall,"max_slip":slip,"pass":z>=12 and m>=18 and f>=.85 and rear>=.80 and fall<=.10 and slip<=.30}
OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(exist_ok=True);masks=json.loads((M0/"a7_environment_masks.json").read_text())["batches"];current=OUT/"checkpoints"/f"model_{args.resume_update:03d}.pt";template=torch.load(current,map_location="cpu",weights_only=False);params=[template["actor_state_dict"][k].clone().requires_grad_() for k in ak]+[template["critic_state_dict"][k].clone().requires_grad_() for k in ck];opt=torch.optim.Adam([{"params":params[:9],"lr":1.5e-5},{"params":params[9:],"lr":1.5e-5}],lr=1.5e-5);opt.load_state_dict(copy.deepcopy(template["optimizer_state_dict"]));std=template["actor_state_dict"]["distribution.log_std_walk"].exp();runtime=copy.deepcopy(template["a7_r2_runtime_state"])
rows=list(csv.DictReader((OUT/"training_curves.csv").open(encoding="utf-8"))) if args.resume_update>1 else []
rows=[x for x in rows if int(x["update"])<=args.resume_update]
manifest=json.loads((OUT/"checkpoint_manifest.json").read_text()) if (OUT/"checkpoint_manifest.json").exists() else {"checkpoints":[]};checks=[x for x in manifest["checkpoints"] if int(x["update"])<=args.resume_update]
if not any(int(x["update"])==args.resume_update for x in checks):checks.append({"update":args.resume_update,"path":str(current.relative_to(OUT)).replace("\\","/"),"sha256":sha(current),"actor_hash":hobj(template["actor_state_dict"]),"critic_hash":hobj(template["critic_state_dict"]),"optimizer_hash":hobj(template["optimizer_state_dict"]),"runtime":copy.deepcopy(runtime)})
guard_doc=json.loads((OUT/"early_guard.json").read_text()) if (OUT/"early_guard.json").exists() else {"rows":[]};guards=[x for x in guard_doc["rows"] if int(x["update"])<=args.resume_update]
if not guards[-1]["pass"]:dump("early_guard.json",{"status":"FAIL","rows":guards});raise SystemExit("EXP013_W2_P1_A7_R2_TRAINING_UNSTABLE early guard")
for update in range(args.resume_update+1,151):
 phase_name,rear_speed=phase(update);policy_hash=hobj([p.detach() for p in params[:9]]);pieces=[];batches=[];offsets=[];allocations=[];unit=RAW/f"update_{update:03d}";unit.mkdir(exist_ok=True);unit_index=0
 while sum(len(x["observation"]) for x in pieces)<24576:
  cursor=runtime["collection_cursor"];batch=cursor%5;offset=OFFSETS[cursor%12];train=torch.tensor(masks[str(batch)]["train_mask"],dtype=torch.bool);ta,tb,alloc,newres=targets(train,rear_speed,cursor,runtime["quota_residual"]);sub=unit/f"unit_{unit_index:02d}";sub.mkdir(exist_ok=True);pa=sub/"targets_a.pt";pb=sub/"targets_b.pt";torch.save(ta,pa);torch.save(tb,pb);oa=sub/"pass_a.pt";ob=sub/"pass_b.pt";run_collector(current,pa,oa,batch,offset,SEED+cursor*2);run_collector(current,pb,ob,batch,offset,SEED+cursor*2);piece,meta=compact_pair((oa,ob));pieces.append(piece);batches.append(batch);offsets.append(offset);allocations.append(alloc);runtime["collection_cursor"]+=1;runtime["quota_residual"]=newres;runtime["teacher_rollin_env_steps"]+=2*(batch+1)*150*N;runtime["prefix_warmup_env_steps"]+=2*offset*N;runtime["total_simulator_env_steps"]+=2*((batch+1)*150+offset+24)*N;unit_index+=1
 storage={k:torch.cat([x[k] for x in pieces]) for k in pieces[0]};metrics=ppo_update(params,opt,storage,update,std);runtime["ppo_interactions"]+=len(storage["observation"]);runtime["update"]=update;current,payload=save_checkpoint(update,params,opt,template,runtime,metrics);row={"update":update,"phase":phase_name,"rear_speed":rear_speed if rear_speed is not None else "MIXED","source_batch":json.dumps(batches),"offset":json.dumps(offsets),"allocation":json.dumps(allocations),"valid_samples":len(storage["observation"]),"ppo_interactions_cumulative":runtime["ppo_interactions"],"teacher_rollin_steps_cumulative":runtime["teacher_rollin_env_steps"],"prefix_warmup_steps_cumulative":runtime["prefix_warmup_env_steps"],"total_simulator_steps_cumulative":runtime["total_simulator_env_steps"],"policy_hash_before":policy_hash,**metrics};rows.append(row)
 if update<=10:
  g=guard(current,update);guards.append(g)
  if not g["pass"]:dump("early_guard.json",{"status":"FAIL","rows":guards});break
 if update in SAVE:checks.append({"update":update,"path":str(current.relative_to(OUT)).replace("\\","/"),"sha256":sha(current),"actor_hash":hobj(payload["actor_state_dict"]),"critic_hash":hobj(payload["critic_state_dict"]),"optimizer_hash":hobj(payload["optimizer_state_dict"]),"runtime":copy.deepcopy(runtime)})
 with (OUT/"training_curves.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 dump("checkpoint_manifest.json",{"checkpoints":checks,"persistent_runs":1});dump("a7_simulator_step_accounting.json",runtime);dump("early_guard.json",{"status":"PASS" if all(x["pass"] for x in guards) else "FAIL","rows":guards});print(json.dumps({"update":update,"phase":phase_name,"batch":batch,"offset":offset,"samples":len(storage["observation"]),"kl":metrics["exact_kl"],"clip":metrics["clip_fraction"],"guard":guards[-1] if update<=10 else "not_applicable"}),flush=True)
 if metrics["nan_inf"] or metrics["exact_kl"]>.5 or (update<=10 and not guards[-1]["pass"]):break
 for sub in unit.glob("unit_*"):
  for p in (sub/"pass_a.pt",sub/"pass_b.pt",sub/"targets_a.pt",sub/"targets_b.pt"):p.unlink(missing_ok=True)
if runtime["update"]<150:dump("stage_classification.json",{"classification":"EXP013_W2_P1_A7_R2_TRAINING_UNSTABLE","stopped_update":runtime["update"]});raise SystemExit(2)
print(json.dumps({"status":"TRAINING_COMPLETE","runtime":runtime},indent=2))
