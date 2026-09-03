"""Formal W2-P1-R2 canonical-parent balanced-only supervised run.

The only policy files written are this stage's 81 scheduled student checkpoints
and its single selected wrapper. Existing datasets/checkpoints are read-only.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, subprocess, sys
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]; sys.path.insert(0,str(HERE.parent))
from train_w2_p1_student import MOVING_GROUPS, Student, load_datasets, sample, split_groups  # noqa:E402

BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
SOURCE=BASE/"phase_w2_p1_practical_stop_endpoint_acquisition"
RESOLVED=BASE/"phase_w2_p1_r1_d2_dataset_provenance_reconciliation"
D3=BASE/"phase_w2_p1_d3_initialization_gap_diagnosis"
OUT=BASE/"phase_w2_p1_r2_long_horizon_group_balanced_stop_integration"
RAW=OUT/"raw"; CKPTS=RAW/"checkpoints"
PARENT=BASE/"phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
PARENT_SHA="61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"
GROUPS=("STOP_RECOVERY","STEADY_STOP",*MOVING_GROUPS,"START_RETENTION")
SAVE_STEPS=(0,*range(500,40001,500)); DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""): h.update(b)
 return h.hexdigest()
def tensor_hash(state):
 h=hashlib.sha256()
 for k in sorted(state):
  v=state[k].detach().cpu().contiguous(); h.update(k.encode()); h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode()); h.update(v.numpy().tobytes())
 return h.hexdigest()
def object_hash(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def dump(name,value): (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def write_csv(name,rows):
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys: keys.append(k)
 with (OUT/name).open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def make_data():
 datasets,groups=load_datasets(); splits=split_groups(datasets,groups); pg=torch.Generator().manual_seed(20276049); vg=torch.Generator().manual_seed(20276100)
 pools={g:sample(g,"train",50000,datasets,splits,pg,DEVICE) for g in GROUPS}; validation={g:sample(g,"validation",5000,datasets,splits,vg,DEVICE) for g in GROUPS}
 return datasets,splits,pools,validation
def draw(group,count,pools):
 v=pools[group]; ids=torch.randint(len(v[0]),(count,),device=DEVICE); return tuple(x[ids] for x in v)
def loss_parts(model,pools):
 def loss(group,count):
  o,g,t=draw(group,count,pools); return nn.functional.mse_loss(model(o,g),t)
 moving=torch.stack([loss(g,64) for g in MOVING_GROUPS]).mean(); stop=loss("STOP_RECOVERY",256); steady=loss("STEADY_STOP",256); start=loss("START_RETENTION",256)
 return .25*(stop+steady+moving+start),{"stop_recovery_loss":float(stop.detach()),"steady_stop_loss":float(steady.detach()),"moving_retention_loss":float(moving.detach()),"start_retention_loss":float(start.detach())}
def evaluate(model,values):
 rows=[]; model.eval()
 with torch.inference_mode():
  for group in GROUPS:
   o,g,t=values[group]; p=model(o,g); e=(p-t).square().mean(1); c=nn.functional.cosine_similarity(p,t,dim=1)
   rows.append({"group":group,"mean_mse":float(e.mean()),"cosine":float(c.mean()),"p95":float(torch.quantile(e,.95)),"p99":float(torch.quantile(e,.99)),"maximum":float(e.max()),"gate_pass":bool(float(e.mean())<=.001 and float(c.mean())>=.98)})
 start_o,start_g,start_t=values["START_RETENTION"]
 with torch.inference_mode(): p=model(start_o,start_g); e=(p-start_t).square().mean(1)
 zero=torch.linalg.vector_norm(start_o[:,9:12],dim=1)==0
 exact={"sample_count":len(e),"exact_zero_count":int(zero.sum()),"exact_zero_mse":float(e[zero].mean()) if zero.any() else None,"nonzero_mse":float(e[~zero].mean())}
 summary={"joint_pass":all(r["gate_pass"] for r in rows),"worst_moving_mse":max(r["mean_mse"] for r in rows if r["group"] in MOVING_GROUPS),"minimum_cosine":min(r["cosine"] for r in rows)}
 return rows,summary,exact
def param_distance(state,parent):
 groups={"input_layer":("first_",),"hidden_layer_2":("hidden.1",),"hidden_layer_3":("hidden.3",),"output_mean_layer":("hidden.5",)}; layer={}
 keys=[k for k in state if not k.startswith("distribution.")]
 total=math.sqrt(sum(float((state[k].detach().cpu().float()-parent[k].detach().cpu().float()).square().sum()) for k in keys))
 for name,prefix in groups.items(): layer[name]=math.sqrt(sum(float((state[k].detach().cpu().float()-parent[k].detach().cpu().float()).square().sum()) for k in keys if k.startswith(prefix)))
 return total,layer
def save(model,opt,step,record,parent_state):
 state={k:v.detach().cpu().clone() for k,v in model.export().items()}; opt_state=opt.state_dict(); payload={"step":step,"actor_state_dict":state,"optimizer_state_dict":opt_state,"record":record,"architecture":[124,256,128,128,37],"std_frozen":True,"parent_sha256":PARENT_SHA,"resolved_manifest":"phase_w2_p1_r1_d2_dataset_provenance_reconciliation/w2_p1_dataset_hashes_resolved_v2.json"}
 path=CKPTS/f"student_step_{step}.pt"; torch.save(payload,path); return {"step":step,"path":str(path.relative_to(REPO)).replace("\\","/"),"sha256":sha(path),"actor_tensor_hash":tensor_hash(state),"optimizer_hash":object_hash(opt_state),"record":record}
def flatten(rows): return {r["group"]:{k:v for k,v in r.items() if k!="group"} for r in rows}

def verify_identity():
 resolved=json.loads((RESOLVED/"w2_p1_dataset_hashes_resolved_v2.json").read_text()); actual={k:sha(REPO/k) for k in resolved["hashes"]}; split=sha(SOURCE/"w2_p1_dataset_split.json"); gate=json.loads((RESOLVED/"dataset_provenance_gate.json").read_text())
 passed=resolved["status"]=="IMMUTABLE_RESOLVED_SOURCE_OF_TRUTH" and actual==resolved["hashes"] and split==resolved["split_sha256"] and gate["next_stage_group_balanced_training_authorized"]
 return {"pass":passed,"manifest":str((RESOLVED/"w2_p1_dataset_hashes_resolved_v2.json").relative_to(REPO)).replace("\\","/"),"status":resolved["status"],"byte_hashes":actual,"semantic_hashes":resolved["semantic_hashes"],"split_expected":resolved["split_sha256"],"split_actual":split,"ordered_sample_identity":"adbf4fd157542ebe602dc75980013779813c6c51179140c87b99e95b9b0e056d"}

def train():
 OUT.mkdir(parents=True,exist_ok=True); CKPTS.mkdir(parents=True,exist_ok=True); identity=verify_identity()
 if not identity["pass"]: raise RuntimeError("DATASET_IDENTITY_MISMATCH")
 if sha(PARENT)!=PARENT_SHA: raise RuntimeError("PARENT_IDENTITY_MISMATCH")
 datasets,splits,pools,validation=make_data(); payload=torch.load(PARENT,map_location="cpu",weights_only=False); parent_state=payload["actor_state_dict"]; model=Student(parent_state).to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=2e-4); initial_norm=float(torch.linalg.vector_norm(torch.cat([p.detach().flatten() for p in model.parameters()])))
 torch.manual_seed(20277717); random.seed(20277717); trace=[]; timeline=[]; manifest=[]; sample_contract={"pool_seed":20276049,"torch_training_seed":20277717,"draw_order":[*MOVING_GROUPS,"STOP_RECOVERY","STEADY_STOP","START_RETENTION"],"draw_sizes":{"moving_each":64,"STOP_RECOVERY":256,"STEADY_STOP":256,"START_RETENTION":256},"pool_size_each":50000}; sample_order_hash=object_hash(sample_contract)
 def checkpoint(step,last):
  rows,summary,exact=evaluate(model,validation); state=model.export(); dist,layers=param_distance(state,parent_state); max_metric=max(r["mean_mse"] for r in rows); finite=all(torch.isfinite(v).all() for v in state.values()); rec={"step":step,"metrics":flatten(rows),"joint_pass":summary["joint_pass"],"worst_moving_mse":summary["worst_moving_mse"],"minimum_cosine":summary["minimum_cosine"],**exact,"parameter_l2_from_parent":dist,"layerwise_parameter_l2":layers,"gradient_norm":last.get("gradient_norm",0.0),"training_loss":last.get("loss"),"training_groups":last.get("parts",{}),"nan_inf":0 if finite else 1}
  timeline.append(rec); manifest.append(save(model,opt,step,rec,parent_state)); print(json.dumps({"step":step,"joint":summary["joint_pass"],"start":rec["metrics"]["START_RETENTION"]["mean_mse"],"stop":rec["metrics"]["STOP_RECOVERY"]["mean_mse"]}),flush=True)
  if not finite or summary["worst_moving_mse"]>.01: raise RuntimeError("NUMERICAL_GUARD")
 checkpoint(0,{})
 prefix=None; last={}
 for step in range(1,40001):
  model.train(); loss,parts=loss_parts(model,pools); opt.zero_grad(set_to_none=True); loss.backward(); grads=[p.grad for p in model.parameters() if p.grad is not None]; finite_grad=all(torch.isfinite(g).all() for g in grads); grad=nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step(); finite_loss=bool(torch.isfinite(loss)); norm=float(torch.linalg.vector_norm(torch.cat([p.detach().flatten() for p in model.parameters()])))
  if not finite_loss or not finite_grad or norm>10*initial_norm: raise RuntimeError("NUMERICAL_GUARD")
  last={"loss":float(loss.detach()),"gradient_norm":float(grad),"parts":parts}
  if step%100==0: trace.append({"step":step,**last})
  if step in SAVE_STEPS:
   checkpoint(step,last)
   if step==2000:
    prefix_trace=[{"step":x["step"],"loss":x["loss"],"gradient_norm":x["gradient_norm"]} for x in trace]
    trace_hash=hashlib.sha256(json.dumps(prefix_trace,sort_keys=True,separators=(",",":")).encode()).hexdigest(); actor_hash=tensor_hash(model.export()); expected=json.loads((D3/"initialization_p3_replay_matrix.json").read_text())["runs"][0]
    current=timeline[-1]; maxdiff=max(abs(current["metrics"][g]["mean_mse"]-expected["final_metrics"][g]["mean_mse"]) for g in GROUPS)
    prefix={"sample_order_contract_hash":sample_order_hash,"sample_order_reference":"D3 contract did not store a distinct sample-index stream hash; tensor+trace are strongest fingerprints","trace_hash":trace_hash,"expected_trace_hash":expected["trace_hash"],"tensor_hash":actor_hash,"expected_tensor_hash":expected["final_tensor_hash"],"maximum_group_metric_difference":maxdiff,"pass":trace_hash==expected["trace_hash"] and actor_hash==expected["final_tensor_hash"] and maxdiff<=1e-8}
    dump("canonical_prefix_parity.json",prefix)
    if not prefix["pass"]: raise RuntimeError("EXP013_W2_P1_R2_PREFIX_PARITY_FAIL")
 write_csv("training_curves.csv",trace); rows=[]
 for t in timeline:
  base={k:v for k,v in t.items() if k!="metrics" and k!="layerwise_parameter_l2" and k!="training_groups"}; base["layerwise_parameter_l2"]=json.dumps(t["layerwise_parameter_l2"],sort_keys=True);base["training_groups"]=json.dumps(t["training_groups"],sort_keys=True)
  for g,v in t["metrics"].items(): base[f"{g}_mse"]=v["mean_mse"];base[f"{g}_cosine"]=v["cosine"];base[f"{g}_pass"]=v["gate_pass"]
  rows.append(base)
 write_csv("validation_checkpoint_timeline.csv",rows); dump("validation_checkpoint_timeline.json",{"checkpoints":timeline,"selection_split":"validation only","held_out_accessed":False})
 candidates=[t for t in timeline if t["joint_pass"]]
 if not candidates:
  dump("checkpoint_manifest.json",{"checkpoints":manifest,"count":len(manifest),"selected":None}); dump("stage_classification.json",{"classification":"EXP013_W2_P1_R2_NO_VALIDATION_JOINT_PASS"}); return
 def rank(t):
  core=max(t["metrics"]["START_RETENTION"]["mean_mse"],t["metrics"]["STOP_RECOVERY"]["mean_mse"],t["metrics"]["STEADY_STOP"]["mean_mse"],t["worst_moving_mse"]); ss=max(t["metrics"]["START_RETENTION"]["mean_mse"],t["metrics"]["STOP_RECOVERY"]["mean_mse"])
  return (core,ss,-t["minimum_cosine"],t["worst_moving_mse"],t["parameter_l2_from_parent"],t["step"])
 selected=min(candidates,key=rank); source=CKPTS/f"student_step_{selected['step']}.pt"; selected_path=RAW/"selected_student.pt"; selected_payload=torch.load(source,map_location="cpu",weights_only=False); selected_payload["selection_rank"]=rank(selected); torch.save(selected_payload,selected_path)
 selected_info={"step":selected["step"],"source_checkpoint":str(source.relative_to(REPO)).replace("\\","/"),"path":str(selected_path.relative_to(REPO)).replace("\\","/"),"sha256":sha(selected_path),"actor_tensor_hash":tensor_hash(selected_payload["actor_state_dict"]),"selection_rank":rank(selected),"validation_metrics":selected["metrics"],"candidate_count":len(candidates),"latest_not_automatically_selected":True,"heldout_used_for_selection":False}; dump("selected_checkpoint.json",selected_info); dump("checkpoint_manifest.json",{"checkpoints":manifest,"count":len(manifest),"selected":selected_info})
 # First and only held-out authorization, after selection.
 hg=torch.Generator().manual_seed(20276023); held={g:sample(g,"held_out",10000,datasets,splits,hg,DEVICE) for g in GROUPS}; sm=Student(selected_payload["actor_state_dict"]).to(DEVICE); hrows,hsummary,hexact=evaluate(sm,held); authorization={"selected_step":selected["step"],"rows":hrows,"summary":hsummary,"exact_zero":hexact,"all_group_pass":hsummary["joint_pass"],"heldout_evaluation_count":1,"checkpoint_fallback_after_heldout":False}; dump("heldout_static_authorization.json",authorization); dump("exact_zero_boundary_results.json",{"validation":{k:selected[k] for k in ("exact_zero_count","exact_zero_mse","nonzero_mse")},"heldout":hexact,"independent_gate":False})
 classification=None if hsummary["joint_pass"] else "EXP013_W2_P1_R2_VALIDATION_SELECTED_HELDOUT_FAIL"; dump("stage_classification.json",{"classification":classification or "PENDING_CLOSED_LOOP","closed_loop_authorized":classification is None})
 dump("resolved_dataset_identity_audit.json",identity); dump("resolved_dataset_hashes.json",{"byte_hashes":identity["byte_hashes"],"semantic_hashes":identity["semantic_hashes"],"split_sha256":identity["split_actual"]}); dump("dataset_identity_gate.json",{"pass":identity["pass"],"dataset_bytes_changed":0,"label_bytes_changed":0,"split_changed":0})
 dump("parent_manifest.json",{"path":str(PARENT.relative_to(REPO)).replace("\\","/"),"sha256":sha(PARENT),"architecture":[124,256,128,128,37],"policy":"W1B-R2 iteration 200"}); dump("parent_identity_audit.json",{"bitwise_initialization":tensor_hash(torch.load(CKPTS/'student_step_0.pt',map_location='cpu',weights_only=False)['actor_state_dict'])==tensor_hash(parent_state),"old_step20k_warm_start_used":False,"mean_actor_updated_only":True,"std_frozen":True,"critic_unused":True})
 dump("numerical_guard.json",{"completed_steps":40000,"nan_inf":0,"parameter_nonfinite":0,"gradient_nonfinite":0,"dataset_identity":"PASS","sample_order":"PASS","parameter_norm_guard":"PASS","moving_validation_guard":"PASS"})
 print(json.dumps({"complete":True,"selected_step":selected["step"],"heldout_pass":hsummary["joint_pass"]}),flush=True)

def parity():
 selected=json.loads((OUT/"selected_checkpoint.json").read_text()); payload=torch.load(REPO/selected["path"],map_location="cpu",weights_only=False); datasets,splits,pools,validation=make_data(); model=Student(payload["actor_state_dict"]).to(DEVICE); replay=Student(payload["actor_state_dict"]).to(DEVICE); rows,summary,exact=evaluate(model,validation); hg=torch.Generator().manual_seed(20276023); held={g:sample(g,"held_out",10000,datasets,splits,hg,DEVICE) for g in GROUPS}; hrows,hsummary,hexact=evaluate(model,held)
 with torch.inference_mode(): same_process_action_difference=max(float((model(validation[g][0],validation[g][1])-replay(validation[g][0],validation[g][1])).abs().max()) for g in GROUPS)
 expected_held=json.loads((OUT/"heldout_static_authorization.json").read_text())["rows"]; maximum=max([abs(r["mean_mse"]-selected["validation_metrics"][r["group"]]["mean_mse"]) for r in rows]+[abs(r["mean_mse"]-e["mean_mse"]) for r,e in zip(hrows,expected_held)])
 passed=tensor_hash(payload["actor_state_dict"])==selected["actor_tensor_hash"] and rows==[{"group":g,**selected["validation_metrics"][g]} for g in GROUPS] and hrows==expected_held and same_process_action_difference<=1e-8
 dump("selected_checkpoint_process_parity.json",{"process":"fresh process relative to training run","parameter_hash":tensor_hash(payload["actor_state_dict"]),"expected_parameter_hash":selected["actor_tensor_hash"],"same_process_double_reload_action_max_difference":same_process_action_difference,"same_process_reload_pass":same_process_action_difference<=1e-8,"serialized_inference_replay_pass":rows==[{"group":g,**selected["validation_metrics"][g]} for g in GROUPS],"validation_metrics":rows,"expected_validation_metrics":selected["validation_metrics"],"heldout_metrics":hrows,"expected_heldout_metrics":expected_held,"maximum_metric_difference":maximum,"pass":passed})

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("mode",choices=("train","parity"));a=p.parse_args();train() if a.mode=="train" else parity()
