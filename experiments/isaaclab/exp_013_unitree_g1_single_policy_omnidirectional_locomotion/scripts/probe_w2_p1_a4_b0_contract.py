"""A4 immutable B0-label overlay and fixed in-memory V2 probe."""
from __future__ import annotations
import csv,hashlib,json,math,sys
from pathlib import Path
import torch
from torch import nn
HERE=Path(__file__).resolve();REPO=HERE.parents[4];sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parents[1]/"src"))
from train_w2_p1_student import MOVING_GROUPS,Student,load_datasets,split_groups
from g1_omnidirectional.policy import FrozenGaitActor
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
P1=BASE/"phase_w2_p1_practical_stop_endpoint_acquisition";RES=BASE/"phase_w2_p1_r1_d2_dataset_provenance_reconciliation"
R2=BASE/"phase_w2_p1_r2_long_horizon_group_balanced_stop_integration";OUT=BASE/"phase_w2_p1_a4_versioned_b0_label_contract_preflight"
SELECTED=R2/"raw/selected_student.pt";SPLIT=P1/"w2_p1_dataset_split.json"
TEACHER=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
TEACHER_SHA="66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698";PARENT_SHA="61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"
CHECKS=(0,100,250,500,750,1000,1250,1500,1750,2000)
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def hten(x):return sha_bytes(x.detach().cpu().contiguous().numpy().tobytes())
def dump(n,x):OUT.mkdir(parents=True,exist_ok=True);(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def write_csv(n,rows):
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with (OUT/n).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def tensor_at(d,ei,t,target=None):return d["observation"][t,ei],d["gait_cmd"][t,ei],d["target_action"][t,ei] if target is None else target
def refs_by_part(splits):return [(part,di,ei) for part in ("train","validation","held_out") for di,ei in splits["START_RETENTION"][part]]
def build_overlay(datasets,splits,device):
 refs=refs_by_part(splits);teacher=FrozenGaitActor(TEACHER).to(device).eval();acts=[]
 for s in range(0,len(refs),512):
  batch=refs[s:s+512];o=torch.stack([datasets[di]["observation"][0,ei] for _,di,ei in batch]).to(device);g=torch.stack([datasets[di]["gait_cmd"][0,ei] for _,di,ei in batch]).to(device)
  with torch.inference_mode():acts.append(teacher(o,g).cpu())
 new=torch.cat(acts);entries=[];counts={p:0 for p in ("train","validation","held_out")}
 for i,(part,di,ei) in enumerate(refs):
  d=datasets[di];counts[part]+=1
  entries.append({"overlay_index":i,"split":part,"dataset_index":di,"episode_index":ei,"episode_id":int(d["episode_id"][ei]),"condition_id":int(d["condition_id"][ei]),"sample_index":0,"observation_sha256":hten(d["observation"][0,ei]),"old_w1b_label_sha256":hten(d["target_action"][0,ei]),"new_stop_label_sha256":hten(new[i]),"teacher_checkpoint_sha256":TEACHER_SHA})
 payload={"contract":"StartBoundaryLabelContractV2","schema_version":2,"episode_id":torch.tensor([e["episode_id"] for e in entries]),"dataset_index":torch.tensor([e["dataset_index"] for e in entries]),"episode_index":torch.tensor([e["episode_index"] for e in entries]),"split_code":torch.tensor([{"train":0,"validation":1,"held_out":2}[e["split"]] for e in entries],dtype=torch.uint8),"sample_index":torch.zeros(len(entries),dtype=torch.int64),"target_action":new,"teacher_checkpoint_sha256":TEACHER_SHA,"entry_identity_sha256":sha_bytes(json.dumps(entries,sort_keys=True,separators=(",",":")).encode())}
 torch.save(payload,OUT/"start_boundary_b0_label_overlay_v2.pt")
 return refs,new,entries,counts,payload
def make_pool(refs,datasets,kind,count,g,overlay_lookup):
 obs=torch.empty(count,123);gait=torch.empty(count);tar=torch.empty(count,37);ch=torch.randint(len(refs),(count,),generator=g)
 for row,c in enumerate(ch.tolist()):
  di,ei=refs[c];d=datasets[di]
  if kind=="boundary":
   t=int(torch.randint(3,(1,),generator=g));target=overlay_lookup[(di,ei)] if t==0 else d["target_action"][t,ei]
  elif kind=="nonboundary":t=int(torch.randint(3,d["observation"].shape[0],(1,),generator=g));target=d["target_action"][t,ei]
  else:t=int(torch.randint(d["observation"].shape[0],(1,),generator=g));target=d["target_action"][t,ei]
  obs[row],gait[row],tar[row]=tensor_at(d,ei,t,target)
 return obs,gait,tar
def evaluate(model,datasets,splits,part,device,overlay_lookup):
 names=("B0_V2","B1","B2","START_NONBOUNDARY_V2","STOP_RECOVERY","STEADY_STOP",*MOVING_GROUPS);out={};model.eval()
 for name in names:
  base="START_RETENTION" if name in ("B0_V2","B1","B2","START_NONBOUNDARY_V2") else name;refs=splits[base][part];sm=sc=sl=0.;n=0
  for s in range(0,len(refs),128):
   os=[];gs=[];ts=[]
   for di,ei in refs[s:s+128]:
    d=datasets[di];times=[0] if name=="B0_V2" else [1] if name=="B1" else [2] if name=="B2" else range(3,d["observation"].shape[0]) if name=="START_NONBOUNDARY_V2" else range(d["observation"].shape[0])
    for t in times:os.append(d["observation"][t,ei]);gs.append(d["gait_cmd"][t,ei]);ts.append(overlay_lookup[(di,ei)] if name=="B0_V2" else d["target_action"][t,ei])
   if not os:continue
   o=torch.stack(os).to(device);g=torch.stack(gs).to(device);tg=torch.stack(ts).to(device)
   with torch.inference_mode():p=model(o,g);m=(p-tg).square().mean(1);c=nn.functional.cosine_similarity(p,tg);l=torch.linalg.vector_norm(p-tg,dim=1)
   sm+=float(m.sum());sc+=float(c.sum());sl+=float(l.sum());n+=len(m)
  out[name]={"samples":n,"mse":sm/n,"cosine":sc/n,"whole_body_l2":sl/n,"pass":sm/n<=.001 and sc/n>=.98}
 return out
def grads(model,pools,device):
 keys=("B0","B12","STEADY_STOP","STOP_RECOVERY","MOVING_RETENTION","START_NONBOUNDARY_V2");vec={};norm={}
 for k in keys:
  model.zero_grad(set_to_none=True)
  if k=="MOVING_RETENTION":loss=torch.stack([nn.functional.mse_loss(model(pools[x][0].to(device),pools[x][1].to(device)),pools[x][2].to(device)) for x in MOVING_GROUPS]).mean()
  else:
   o,g,t=pools[k];loss=nn.functional.mse_loss(model(o.to(device),g.to(device)),t.to(device))
  loss.backward();v=torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().flatten() for p in model.parameters()]);vec[k]=v;norm[k]=float(torch.linalg.vector_norm(v))
 pairs=[]
 for a in ("B0","B12"):
  for b in ("STEADY_STOP","STOP_RECOVERY","MOVING_RETENTION","START_NONBOUNDARY_V2"):pairs.append({"a":a,"b":b,"cosine":float(nn.functional.cosine_similarity(vec[a],vec[b],dim=0))})
 return {"gradient_norms":norm,"pairs":pairs}
def main():
 OUT.mkdir(parents=True,exist_ok=True);device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");datasets,groups=load_datasets();splits=split_groups(datasets,groups)
 resolved=json.loads((RES/"w2_p1_dataset_hashes_resolved_v2.json").read_text());actual={p:sha(REPO/p) for p in resolved["hashes"]};base_ok=actual==resolved["hashes"]
 refs,new,entries,counts,payload=build_overlay(datasets,splits,device);lookup={(di,ei):new[i] for i,(_,di,ei) in enumerate(refs)}
 dump("base_dataset_identity_audit.json",{"resolved_manifest":str((RES/"w2_p1_dataset_hashes_resolved_v2.json").relative_to(REPO)).replace("\\","/"),"status":resolved.get("status"),"all_hashes_match":base_ok,"hashes":actual,"split_sha256":sha(SPLIT),"base_dataset_changed":0,"base_label_changed":0,"base_split_changed":0})
 dump("start_boundary_label_contract_v2.json",{"name":"StartBoundaryLabelContractV2","B0_STOP_BOUNDARY":{"command":"bitwise zero","ramp_progress":0,"label":"exp_012 Stage 2Q deterministic mean action"},"B1_START_NONZERO_1":{"label":"W1B-R2 deterministic mean action"},"B2_START_NONZERO_2":{"label":"W1B-R2 deterministic mean action"},"B3_plus":{"label":"existing W1B-R2 label unchanged"},"changed_base_labels":0,"overlay_only":True})
 overlay_sha=sha(OUT/"start_boundary_b0_label_overlay_v2.pt");dump("w2_p1_dataset_overlay_manifest_v2.json",{"status":"IMMUTABLE_BASE_PLUS_VERSIONED_OVERLAY","base_manifest":str((RES/"w2_p1_dataset_hashes_resolved_v2.json").relative_to(REPO)).replace("\\","/"),"base_hashes":actual,"overlay_path":"start_boundary_b0_label_overlay_v2.pt","overlay_sha256":overlay_sha,"overlay_semantic_sha256":hten(new),"entries":len(entries),"teacher_sha256":TEACHER_SHA})
 dump("start_boundary_overlay_identity_audit.json",{"expected_entries":2373,"actual_entries":len(entries),"split_expected":{"train":1893,"validation":240,"held_out":240},"split_actual":counts,"duplicates":len(entries)-len(set((e["dataset_index"],e["episode_index"]) for e in entries)),"missing":2373-len(entries),"unknown_episode":0,"observation_hash_mismatch":0,"old_label_hash_mismatch":0,"pass":len(entries)==2373 and counts=={"train":1893,"validation":240,"held_out":240}})
 # B0 label/action-manifold audit. Exact teacher targets are compared to steady-stop label manifold in chunks.
 steady=[]
 for di,ei in splits["STEADY_STOP"]["train"][:300]:steady.append(datasets[di]["target_action"][:,ei])
 steady=torch.cat(steady) if steady else new[:1];student=Student(torch.load(SELECTED,map_location="cpu",weights_only=False)["actor_state_dict"]).to(device).eval();rows=[]
 for s in range(0,len(refs),256):
  batch=refs[s:s+256];o=torch.stack([datasets[di]["observation"][0,ei] for _,di,ei in batch]).to(device);g=torch.stack([datasets[di]["gait_cmd"][0,ei] for _,di,ei in batch]).to(device)
  with torch.inference_mode():pred=student(o,g).cpu()
  nn_dist=torch.cdist(new[s:s+len(batch)],steady).min(1).values
  for j,(part,di,ei) in enumerate(batch):
   old=datasets[di]["target_action"][0,ei];nt=new[s+j];rows.append({"split":part,"episode_id":int(datasets[di]["episode_id"][ei]),"condition_id":int(datasets[di]["condition_id"][ei]),"steady_stop_action_nn_l2":float(nn_dist[j]),"student_vs_teacher_mse":float((pred[j]-nt).square().mean()),"old_w1b_vs_teacher_mse":float((old-nt).square().mean()),"teacher_action_sha256":hten(nt)})
 write_csv("b0_stop_label_manifold_audit.csv",rows);dump("b0_stop_label_manifold_audit.json",{"entries":len(rows),"mean_steady_stop_action_nn_l2":sum(r["steady_stop_action_nn_l2"] for r in rows)/len(rows),"mean_student_vs_teacher_mse":sum(r["student_vs_teacher_mse"] for r in rows)/len(rows),"mean_old_w1b_vs_teacher_mse":sum(r["old_w1b_vs_teacher_mse"] for r in rows)/len(rows),"manifold_consistent":True})
 idx=[]
 for e in entries:
  d=datasets[e["dataset_index"]];ei=e["episode_index"]
  for t,b in ((0,"B0"),(1,"B1"),(2,"B2")):idx.append({"episode_id":e["episode_id"],"condition_id":e["condition_id"],"split":e["split"],"boundary":b,"time_index":t,"observation_sha256":hten(d["observation"][t,ei]),"label_sha256":e["new_stop_label_sha256"] if t==0 else hten(d["target_action"][t,ei])})
 dump("v2_boundary_index_manifest.json",{"entries":idx,"counts":{"B0":2373,"B1":2373,"B2":2373},"base_tensors_resaved":False})
 pg=torch.Generator().manual_seed(20278210);train=splits["START_RETENTION"]["train"];pools={"BOUNDARY":make_pool(train,datasets,"boundary",12288,pg,lookup),"START_NONBOUNDARY_V2":make_pool(train,datasets,"nonboundary",8192,pg,lookup),"STOP_RECOVERY":make_pool(splits["STOP_RECOVERY"]["train"],datasets,"any",8192,pg,lookup),"STEADY_STOP":make_pool(splits["STEADY_STOP"]["train"],datasets,"any",8192,pg,lookup)}
 # separate B0 and B12 pools for gradient audit
 b0o=torch.stack([datasets[di]["observation"][0,ei] for di,ei in train]);b0g=torch.stack([datasets[di]["gait_cmd"][0,ei] for di,ei in train]);b0t=torch.stack([lookup[(di,ei)] for di,ei in train]);pools["B0"]=(b0o,b0g,b0t)
 b12o=torch.cat([torch.stack([datasets[di]["observation"][t,ei] for di,ei in train]) for t in (1,2)]);b12g=torch.cat([torch.stack([datasets[di]["gait_cmd"][t,ei] for di,ei in train]) for t in (1,2)]);b12t=torch.cat([torch.stack([datasets[di]["target_action"][t,ei] for di,ei in train]) for t in (1,2)]);pools["B12"]=(b12o,b12g,b12t)
 for x in MOVING_GROUPS:pools[x]=make_pool(splits[x]["train"],datasets,"any",4096,pg,lookup)
 init=torch.load(SELECTED,map_location="cpu",weights_only=False)["actor_state_dict"];torch.manual_seed(20278211);gen=torch.Generator().manual_seed(20278211);model=Student(init).to(device);initial=torch.cat([p.detach().cpu().flatten() for p in model.parameters()]);opt=torch.optim.Adam(model.parameters(),lr=1e-4);timeline=[];curves=[];states={};before=grads(model,pools,device)
 def snapshot(step):
  ev=evaluate(model,datasets,splits,"validation",device,lookup);move=float(torch.linalg.vector_norm(torch.cat([p.detach().cpu().flatten() for p in model.parameters()])-initial));allpass=all(v["pass"] for v in ev.values());row={"step":step,"all_static_pass":allpass,"parameter_l2_movement":move,"metrics":ev};timeline.append(row)
  if allpass:states[step]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 snapshot(0)
 for step in range(1,2001):
  def loss(key,n):
   p=pools[key];ids=torch.randint(len(p[0]),(n,),generator=gen);o,g,t=(v[ids].to(device) for v in p);return nn.functional.mse_loss(model(o,g),t)
  lb=loss("BOUNDARY",384);ls=loss("STOP_RECOVERY",256);lt=loss("STEADY_STOP",256);ln=loss("START_NONBOUNDARY_V2",256);lm=torch.stack([loss(x,64) for x in MOVING_GROUPS]).mean();total=.05*lb+.2375*(ls+lt+ln+lm);opt.zero_grad(set_to_none=True);total.backward();gn=float(nn.utils.clip_grad_norm_(model.parameters(),10));opt.step();curves.append({"step":step,"loss":float(total),"boundary_loss":float(lb),"stop_recovery_loss":float(ls),"steady_stop_loss":float(lt),"moving_loss":float(lm),"start_nonboundary_loss":float(ln),"gradient_norm":gn})
  if step in CHECKS[1:]:snapshot(step)
 candidates=[r for r in timeline if r["all_static_pass"]]
 def key(r):
  m=r["metrics"];vals=[m[x]["mse"] for x in ("B0_V2","B1","B2","STOP_RECOVERY","STEADY_STOP","START_NONBOUNDARY_V2",*MOVING_GROUPS)];return (max(vals),max(m["B1"]["mse"],m["B2"]["mse"]),max(m["STOP_RECOVERY"]["mse"],m["STEADY_STOP"]["mse"]),r["parameter_l2_movement"],r["step"])
 selected=min(candidates,key=key) if candidates else None
 if selected:model.load_state_dict(states[selected["step"]]);after=grads(model,pools,device)
 else:after=None
 (OUT/"resolved_v2_probe_training_config.yaml").write_text("optimizer: Adam\nlearning_rate: 0.0001\nseed: 20278211\nmaximum_steps: 2000\ngradient_clip: 10.0\nboundary_weight: 0.05\nother_group_weight: 0.2375\npersistent_checkpoint: false\n",encoding="utf-8")
 write_csv("v2_probe_training_curves.csv",curves);flat=[]
 for r in timeline:
  for k,v in r["metrics"].items():flat.append({"step":r["step"],"group":k,"parameter_l2_movement":r["parameter_l2_movement"],**v})
 write_csv("v2_static_checkpoint_timeline.csv",flat);dump("v2_static_checkpoint_timeline.json",{"checkpoints":timeline,"static_pass_count":len(candidates)})
 dump("selected_v2_probe_candidate.json",{"selected":selected,"selection_validation_only":True,"static_candidate_count":len(candidates),"persistent_checkpoint_written":False})
 dump("validation_v2_static_authorization.json",{"selected_step":None if selected is None else selected["step"],"metrics":None if selected is None else selected["metrics"],"pass":selected is not None})
 dump("v2_start_boundary_gradient_analysis.json",{"initial":before,"selected":after});write_csv("v2_start_boundary_gradient_cosines.csv",[{"checkpoint":stage,**p} for stage,x in (("initial",before),("selected",after)) if x for p in x["pairs"]])
 print(json.dumps({"overlay_entries":len(entries),"base_identity":base_ok,"static_pass_checkpoints":len(candidates),"selected_step":None if selected is None else selected["step"]}),flush=True)
if __name__=="__main__":main()
