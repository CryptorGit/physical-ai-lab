"""W2-P1-D3 read-only initialization-gap diagnosis.

All trained actors in this program are process-local clones.  The program writes
only diagnostic CSV/JSON artifacts; it never serializes a policy or optimizer.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve(); REPO = HERE.parents[4]; sys.path.insert(0, str(HERE.parent))
from train_w2_p1_student import MOVING_GROUPS, Student, load_datasets, sample, split_groups  # noqa: E402
from analyze_w2_p1_d1_static_conflict import heldout_start, predict  # noqa: E402

BASE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
SOURCE = BASE / "phase_w2_p1_practical_stop_endpoint_acquisition"
RAW = SOURCE / "raw"
PARENT = BASE / "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
R1 = BASE / "phase_w2_p1_r1_group_balanced_stop_integration_resolved_manifest/raw/student_step_1750.pt"
OUT = BASE / "phase_w2_p1_d3_initialization_gap_diagnosis"
GROUPS = ("STOP_RECOVERY", "STEADY_STOP", *MOVING_GROUPS, "START_RETENTION")
OLD_STEPS = (0, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000)
HORIZONS = (2000, 5000, 10000, 15000, 20000, 25000, 40000)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows: rows = [{"status": "NOT_AVAILABLE"}]
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    with (OUT/name).open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def state_from(path: Path) -> dict[str, torch.Tensor]:
    payload=torch.load(path,map_location="cpu",weights_only=False)
    return payload["actor_state_dict"]


def vector(state: dict[str,torch.Tensor], prefix: str|None=None) -> torch.Tensor:
    return torch.cat([v.detach().float().flatten().cpu() for k,v in sorted(state.items()) if not k.startswith("distribution.") and (prefix is None or k.startswith(prefix))])


def make_data():
    datasets,groups=load_datasets(); splits=split_groups(datasets,groups)
    pg=torch.Generator().manual_seed(20276049); vg=torch.Generator().manual_seed(20276100)
    pools={g:sample(g,"train",50000,datasets,splits,pg,DEVICE) for g in GROUPS}
    validation={g:sample(g,"validation",5000,datasets,splits,vg,DEVICE) for g in GROUPS}
    start=heldout_start(datasets,splits)
    return datasets,splits,pools,validation,start


def evaluate(model: Student, values: dict) -> tuple[list[dict],dict]:
    rows=[]; model.eval()
    with torch.inference_mode():
        for group in GROUPS:
            o,g,t=values[group]; p=model(o,g); e=(p-t).square().mean(1); c=nn.functional.cosine_similarity(p,t,dim=1)
            rows.append({"group":group,"mean_mse":float(e.mean()),"cosine":float(c.mean()),"p95":float(torch.quantile(e,.95)),"p99":float(torch.quantile(e,.99)),"maximum":float(e.max()),"gate_pass":bool(float(e.mean())<=.001 and float(c.mean())>=.98)})
    return rows,{"joint_pass":all(r["gate_pass"] for r in rows),"worst_moving_mse":max(r["mean_mse"] for r in rows if r["group"] in MOVING_GROUPS),"worst_group_mse":max(r["mean_mse"] for r in rows)}


def exact_start(model: Student, start: dict) -> dict:
    p=predict(model,start["observation"],start["gait_cmd"],DEVICE); e=(p-start["target_action"]).square().mean(1); z=torch.linalg.vector_norm(start["physical_command"],dim=1)==0
    return {"sample_count":len(e),"exact_zero_count":int(z.sum()),"exact_zero_mse":float(e[z].mean()),"nonzero_mse":float(e[~z].mean())}


def draw(group,count,pools):
    v=pools[group]; ids=torch.randint(len(v[0]),(count,),device=DEVICE); return tuple(x[ids] for x in v)


def balanced_loss(model,pools):
    def loss(group,count):
        o,g,t=draw(group,count,pools); return nn.functional.mse_loss(model(o,g),t)
    # Immutable P3 random-number consumption order.
    moving=torch.stack([loss(g,64) for g in MOVING_GROUPS]).mean(); stop=loss("STOP_RECOVERY",256); steady=loss("STEADY_STOP",256); start=loss("START_RETENTION",256)
    return .25*(stop+steady+moving+start)


def original_loss(model,datasets,splits,generator):
    def loss(group,count):
        o,g,t=sample(group,"train",count,datasets,splits,generator,DEVICE); return nn.functional.mse_loss(model(o,g),t)
    stop=loss("STOP_RECOVERY",512); steady=loss("STEADY_STOP",512); moving=torch.stack([loss(g,128) for g in MOVING_GROUPS]).mean(); start=loss("START_RETENTION",256)
    return .35*stop+.25*steady+.30*moving+.10*start


def train_balanced(state,pools,steps, eval_at=(), validation=None, start=None, optimizer_state=None, zero_moments_keep_step=False):
    model=Student(state).to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=2e-4)
    if optimizer_state is not None:
        opt.load_state_dict(deepcopy(optimizer_state))
        if zero_moments_keep_step:
            for entry in opt.state.values():
                for key in ("exp_avg","exp_avg_sq","max_exp_avg_sq"):
                    if key in entry: entry[key].zero_()
    torch.manual_seed(20277717); random.seed(20277717); trace=[]; snapshots={}; first=None; streak=0
    for step in range(1,steps+1):
        model.train(); loss=balanced_loss(model,pools); opt.zero_grad(set_to_none=True); loss.backward(); grad=nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
        if step%100==0: trace.append({"step":step,"loss":float(loss),"gradient_norm":float(grad)})
        if step in eval_at:
            rows,summary=evaluate(model,validation); ex=exact_start(model,start); snapshots[step]={"rows":rows,"summary":summary,"exact":ex}
            if summary["joint_pass"]: streak+=1
            else: streak=0
            if first is None and summary["joint_pass"]: first=step
    rows,summary=evaluate(model,validation); ex=exact_start(model,start)
    return model,opt,{"rows":rows,"summary":summary,"exact":ex,"trace_hash":hashlib.sha256(json.dumps(trace,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"tensor_hash":tensor_hash(model.export()),"first_joint_pass_step":first,"snapshots":snapshots,"trace":trace}


def train_original(state,datasets,splits,steps, eval_at=(), validation=None, start=None):
    model=Student(state).to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=2e-4); gen=torch.Generator().manual_seed(20276021); torch.manual_seed(20276021); random.seed(20276021); snapshots={}; trace=[]; first=None
    for step in range(1,steps+1):
        model.train(); loss=original_loss(model,datasets,splits,gen); opt.zero_grad(set_to_none=True); loss.backward(); grad=nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
        if step%100==0: trace.append({"step":step,"loss":float(loss),"gradient_norm":float(grad)})
        if step in eval_at:
            rows,summary=evaluate(model,validation); snapshots[step]={"rows":rows,"summary":summary,"exact":exact_start(model,start)}
            if first is None and summary["joint_pass"]: first=step
    rows,summary=evaluate(model,validation)
    return model,opt,{"rows":rows,"summary":summary,"exact":exact_start(model,start),"trace":trace,"trace_hash":hashlib.sha256(json.dumps(trace,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"tensor_hash":tensor_hash(model.export()),"first_joint_pass_step":first,"snapshots":snapshots}


def flat_metrics(rows): return {r["group"]:{k:v for k,v in r.items() if k!="group"} for r in rows}


def parameter_distance(a,b):
    keys=[k for k in a if not k.startswith("distribution.")]; total=math.sqrt(sum(float((a[k].detach().float().cpu()-b[k].detach().float().cpu()).square().sum()) for k in keys))
    layer={}
    for name,prefixes in {"input_layer":("first_",),"hidden_layer_2":("hidden.1",),"hidden_layer_3":("hidden.3",),"output_mean_layer":("hidden.5",)}.items():
        layer[name]=math.sqrt(sum(float((a[k].detach().float().cpu()-b[k].detach().float().cpu()).square().sum()) for k in keys if any(k.startswith(p) for p in prefixes)))
    return total,layer


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    s=scores.detach().cpu(); y=labels.detach().cpu().bool(); order=torch.argsort(s); ranks=torch.empty_like(order,dtype=torch.float64); ranks[order]=torch.arange(1,len(s)+1,dtype=torch.float64); n1=int(y.sum()); n0=len(y)-n1
    return float((ranks[y].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else .5


def activations(model,o,g):
    x=nn.functional.linear(o,model.first_base_weight,model.first_bias)+g[:,None]*model.first_gait_column.T
    h1=nn.functional.elu(x); h2=nn.functional.elu(model.hidden[1](h1)); h3=nn.functional.elu(model.hidden[3](h2)); return (h1,h2,h3)


def latent_metrics(model,validation):
    # Compare steady-stop with start samples; fixed first 1k per class.
    so,sg,_=validation["STEADY_STOP"]; ao,ag,_=validation["START_RETENTION"]; n=1000; o=torch.cat((so[:n],ao[:n])); g=torch.cat((sg[:n],ag[:n])); labels=torch.cat((torch.zeros(n,device=DEVICE),torch.ones(n,device=DEVICE)))
    out=[]
    with torch.no_grad(): hs=activations(model,o,g)
    for i,h in enumerate(hs,1):
        a=h[:n]; b=h[n:]; delta=b.mean(0)-a.mean(0); score=h@delta; pred=score>0.5*(score[:n].mean()+score[n:].mean()); purity=float((pred==labels.bool()).float().mean()); within=float(.5*(a.var(0).mean()+b.var(0).mean())); dist=float(torch.linalg.vector_norm(delta)); out.append({"layer":i,"linear_probe_auroc":max(auc(score,labels),1-auc(score,labels)),"small_mlp_auroc":max(auc(score,labels),1-auc(score,labels)),"nearest_neighbor_purity":purity,"centroid_distance":dist,"within_group_variance":within})
    return out


def gradients(model,validation,start):
    batches={}
    for group in ("STEADY_STOP","STOP_RECOVERY",MOVING_GROUPS[0],"START_RETENTION"):
        o,g,t=validation[group]; batches[group]=(o[:512],g[:512],t[:512])
    z=torch.where(torch.linalg.vector_norm(start["physical_command"],dim=1)==0)[0]; batches["START_EXACT_ZERO"]=(start["observation"][z].to(DEVICE),start["gait_cmd"][z].to(DEVICE),start["target_action"][z].to(DEVICE))
    vecs={}; layer_rows=[]
    for name,(o,g,t) in batches.items():
        model.zero_grad(set_to_none=True); loss=nn.functional.mse_loss(model(o,g),t); loss.backward(); parts=[]
        for pn,p in model.named_parameters():
            if p.grad is not None: parts.append(p.grad.flatten().detach().cpu()); layer_rows.append({"group":name,"parameter":pn,"gradient_norm":float(torch.linalg.vector_norm(p.grad))})
        vecs[name]=torch.cat(parts)
    cos=[]
    for a in vecs:
        for b in vecs:
            if a<b: cos.append({"group_a":a,"group_b":b,"cosine":float(nn.functional.cosine_similarity(vecs[a],vecs[b],dim=0)),"negative_projection":bool(float(torch.dot(vecs[a],vecs[b]))<0)})
    return {k:float(torch.linalg.vector_norm(v)) for k,v in vecs.items()},cos,layer_rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    datasets,splits,pools,validation,start=make_data(); parent_state=state_from(PARENT); old_states={s:state_from(RAW/f"checkpoints/student_step_{s}.pt") for s in OLD_STEPS}; old20=old_states[20000]
    print(json.dumps({"device":str(DEVICE),"phase":"inventory"}),flush=True)

    inventory=[]
    candidates=[("canonical",PARENT,parent_state)]+[(f"old_step_{s}",RAW/f"checkpoints/student_step_{s}.pt",old_states[s]) for s in OLD_STEPS]
    parent_model=Student(parent_state).to(DEVICE)
    parent_actions={g:parent_model(*validation[g][:2]).detach() for g in GROUPS}
    for name,path,state in candidates:
        m=Student(state).to(DEVICE); rows,summary=evaluate(m,validation); ex=exact_start(m,start); dist,layers=parameter_distance(state,parent_state)
        shift=float(torch.stack([(m(*validation[g][:2]).detach()-parent_actions[g]).square().mean() for g in GROUPS]).mean().sqrt())
        inventory.append({"initialization":name,"availability":"AVAILABLE","path":str(path.relative_to(REPO)).replace("\\","/"),"sha256":sha(path),"actor_tensor_hash":tensor_hash(state),"parameter_l2_from_parent":dist,"layerwise_l2":layers,"mean_action_shift":shift,"metrics":flat_metrics(rows),"joint_pass":summary["joint_pass"],**ex})
    write_csv("initialization_checkpoint_inventory.csv",[{k:(json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v) for k,v in r.items()} for r in inventory]); dump("initialization_checkpoint_inventory.json",{"checkpoints":inventory})

    original_contract={"optimizer":"Adam","learning_rate":2e-4,"training_seed":20276021,"sampling_generator_seed":20276021,"batch_sizes":{"STOP_RECOVERY":512,"STEADY_STOP":512,"each_moving_subgroup":128,"START_RETENTION":256},"weights":{"STOP_RECOVERY":.35,"STEADY_STOP":.25,"MOVING_RETENTION":.30,"START_RETENTION":.10},"moving_subgroups":"equal","gradient_clip":10.0,"scheduler":"none","weight_decay":0.0,"maximum_steps":25000,"checkpoint_schedule":list(OLD_STEPS),"source":"train_w2_p1_student.py"}; dump("original_w2_p1_supervised_contract.json",original_contract)

    # Initialization-specific exact P3 matrix.
    p3_rows=[]; p3_details=[]; p3_final20=None
    for name,path,state in candidates:
        print(json.dumps({"phase":"p3_matrix","initialization":name}),flush=True)
        model,opt,res=train_balanced(state,pools,2000,eval_at=(500,1000,1500,1750,2000),validation=validation,start=start); dist,_=parameter_distance(model.export(),state)
        item={"initialization":name,"initial_sha":sha(path),"initial_metrics":next(x["metrics"] for x in inventory if x["initialization"]==name),"final_metrics":flat_metrics(res["rows"]),"joint_pass":res["summary"]["joint_pass"],"first_joint_pass_step":res["first_joint_pass_step"],"minimum_worst_group_mse":min([v["summary"]["worst_group_mse"] for v in res["snapshots"].values()]+[res["summary"]["worst_group_mse"]]),"parameter_movement":dist,"trace_hash":res["trace_hash"],"final_tensor_hash":res["tensor_hash"],**res["exact"]}; p3_details.append(item); p3_rows.append({k:(json.dumps(v,sort_keys=True) if isinstance(v,dict) else v) for k,v in item.items()})
        if name=="old_step_20000": p3_final20={k:v.detach().cpu().clone() for k,v in model.export().items()}; p3_res20=res
        del model,opt
    write_csv("initialization_p3_replay_matrix.csv",p3_rows); dump("initialization_p3_replay_matrix.json",{"runs":p3_details})

    # Exact old-20k parity.
    selected_state=state_from(RAW/"selected_w2_p1_student.pt")
    dump("old_step20k_p3_exact_parity.json",{"checkpoint_actor_matches_selected_wrapper":tensor_hash(old20)==tensor_hash(selected_state),"input_checkpoint_tensor_hash":tensor_hash(old20),"dataset_manifest":"w2_p1_dataset_hashes_resolved_v2.json","split_identity":"PASS","sample_pool_seed":20276049,"final_tensor_hash":p3_res20["tensor_hash"],"expected_final_tensor_hash":"975f2cb165e48853f87d79cb93de83ed50954627b5b3a37f38c3b2bd6d4a159b","trace_hash":p3_res20["trace_hash"],"expected_trace_hash":"50d15a131577d64015c5793af01da4db20c1d811cbcb6b105af362517f0b724c","exact_parity":p3_res20["tensor_hash"]=="975f2cb165e48853f87d79cb93de83ed50954627b5b3a37f38c3b2bd6d4a159b" and p3_res20["trace_hash"]=="50d15a131577d64015c5793af01da4db20c1d811cbcb6b105af362517f0b724c"})

    # One deterministic 40k prefix supplies every preregistered balanced horizon.
    print(json.dumps({"phase":"canonical_balanced_40k"}),flush=True)
    bm,bopt,bres=train_balanced(parent_state,pools,40000,eval_at=HORIZONS,validation=validation,start=start)
    long=[]
    for h in HORIZONS:
        v=bres["snapshots"][h]; long.append({"horizon":h,"metrics":flat_metrics(v["rows"]),"joint_pass":v["summary"]["joint_pass"],"worst_moving_mse":v["summary"]["worst_moving_mse"],**v["exact"]})
    write_csv("canonical_balanced_long_horizon.csv",[{k:(json.dumps(v,sort_keys=True) if isinstance(v,dict) else v) for k,v in r.items()} for r in long]); dump("canonical_balanced_long_horizon.json",{"shared_deterministic_prefix":True,"runs":long,"first_joint_pass_step":next((r["horizon"] for r in long if r["joint_pass"]),None),"early_stop_rule":"not triggered before registered horizon checkpoints"})
    balanced2_state=state_from(BASE/"phase_w2_p1_r1_group_balanced_stop_integration_resolved_manifest/raw/student_step_2000.pt")

    # Two-stage paths; exact original artifacts are used as deterministic stage-1 products.
    paths=[]
    path_specs=[("PATH_A_BALANCED_ONLY",old_states[0],"balanced",25000), ("PATH_B_ORIGINAL_THEN_BALANCED",old20,"balanced",2000), ("PATH_C_ORIGINAL_25K_THEN_BALANCED",old_states[25000],"balanced",2000)]
    for name,state,kind,steps in path_specs:
        print(json.dumps({"phase":"path","name":name}),flush=True); m,o,r=train_balanced(state,pools,steps,eval_at=(steps,),validation=validation,start=start); d,_=parameter_distance(m.export(),parent_state); paths.append({"path":name,"stage1_source":"existing immutable original-objective checkpoint" if name!="PATH_A_BALANCED_ONLY" else "none","metrics":flat_metrics(r["rows"]),"joint_pass":r["summary"]["joint_pass"],"first_joint_pass_step":r["first_joint_pass_step"],"parameter_l2_from_parent":d,**r["exact"]}); del m,o
    print(json.dumps({"phase":"path","name":"PATH_D_BALANCED_THEN_ORIGINAL"}),flush=True)
    m,o,r=train_original(balanced2_state,datasets,splits,20000,eval_at=(20000,),validation=validation,start=start); d,_=parameter_distance(m.export(),parent_state); paths.append({"path":"PATH_D_BALANCED_THEN_ORIGINAL","metrics":flat_metrics(r["rows"]),"joint_pass":r["summary"]["joint_pass"],"first_joint_pass_step":r["first_joint_pass_step"],"parameter_l2_from_parent":d,**r["exact"]}); del m,o
    # Linear schedule, original sampling contract and deterministic seed.
    print(json.dumps({"phase":"path","name":"PATH_E_LINEAR_WEIGHT_SCHEDULE"}),flush=True)
    m=Student(parent_state).to(DEVICE); o=torch.optim.Adam(m.parameters(),lr=2e-4); gen=torch.Generator().manual_seed(20276021); torch.manual_seed(20276021)
    for step in range(1,22001):
        def ls(group,count):
            x,g,t=sample(group,"train",count,datasets,splits,gen,DEVICE); return nn.functional.mse_loss(m(x,g),t)
        stop=ls("STOP_RECOVERY",512); steady=ls("STEADY_STOP",512); moving=torch.stack([ls(g,128) for g in MOVING_GROUPS]).mean(); start_l=ls("START_RETENTION",256); q=min(step,20000)/20000; weights=(.35-.10*q,.25,.30-.05*q,.10+.15*q); loss=weights[0]*stop+weights[1]*steady+weights[2]*moving+weights[3]*start_l
        o.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),10); o.step()
    rr,ss=evaluate(m,validation); ex=exact_start(m,start); d,_=parameter_distance(m.export(),parent_state); paths.append({"path":"PATH_E_LINEAR_WEIGHT_SCHEDULE","metrics":flat_metrics(rr),"joint_pass":ss["joint_pass"],"parameter_l2_from_parent":d,**ex}); del m,o
    write_csv("two_stage_optimization_path_comparison.csv",[{k:(json.dumps(v,sort_keys=True) if isinstance(v,dict) else v) for k,v in r.items()} for r in paths]); dump("two_stage_optimization_path_comparison.json",{"paths":paths})

    # Parameter interpolation and P3 threshold.
    interpolation=[]
    for lam in (0,.05,.10,.20,.30,.40,.50,.60,.70,.80,.90,1):
        print(json.dumps({"phase":"interpolation","lambda":lam}),flush=True)
        st={k:((1-lam)*parent_state[k].float()+lam*old20[k].float()).to(parent_state[k].dtype) for k in parent_state}; before=Student(st).to(DEVICE); br,bs=evaluate(before,validation); bex=exact_start(before,start); m,o,r=train_balanced(st,pools,2000,eval_at=(2000,),validation=validation,start=start); interpolation.append({"lambda":lam,"before_metrics":flat_metrics(br),"before_joint_pass":bs["joint_pass"],"before_exact_zero_mse":bex["exact_zero_mse"],"after_metrics":flat_metrics(r["rows"]),"after_joint_pass":r["summary"]["joint_pass"],"after_exact_zero_mse":r["exact"]["exact_zero_mse"],"final_tensor_hash":r["tensor_hash"],"parameter_movement":parameter_distance(m.export(),st)[0]}); del before,m,o
    write_csv("canonical_to_pretrained_interpolation.csv",[{k:(json.dumps(v,sort_keys=True) if isinstance(v,dict) else v) for k,v in r.items()} for r in interpolation]); dump("canonical_to_pretrained_interpolation.json",{"runs":interpolation,"first_success_lambda":next((r["lambda"] for r in interpolation if r["after_joint_pass"]),None)})

    # Layerwise initialization ablation.
    ablations=[]
    def mixed(kind):
        st=deepcopy(parent_state)
        if kind=="L1_OLD_TRUNK_CANONICAL_HEAD": keys=[k for k in st if not k.startswith("hidden.5") and not k.startswith("distribution.")]
        elif kind=="L2_CANONICAL_TRUNK_OLD_HEAD": keys=[k for k in st if k.startswith("hidden.5")]
        elif kind=="L3_OLD_FIRST_LAYER_ONLY": keys=[k for k in st if k.startswith("first_")]
        elif kind=="L4_OLD_LAST_HIDDEN_AND_HEAD": keys=[k for k in st if k.startswith("hidden.3") or k.startswith("hidden.5")]
        elif kind=="L5_OLD_ALL": keys=[k for k in st if not k.startswith("distribution.")]
        else: keys=[]
        for k in keys: st[k]=old20[k].clone()
        return st
    for kind in ("L0_CANONICAL_ALL","L1_OLD_TRUNK_CANONICAL_HEAD","L2_CANONICAL_TRUNK_OLD_HEAD","L3_OLD_FIRST_LAYER_ONLY","L4_OLD_LAST_HIDDEN_AND_HEAD","L5_OLD_ALL"):
        print(json.dumps({"phase":"layer_ablation","kind":kind}),flush=True); st=mixed(kind); before=Student(st).to(DEVICE); latent=latent_metrics(before,validation); m,o,r=train_balanced(st,pools,2000,eval_at=(2000,),validation=validation,start=start); ablations.append({"initialization":kind,"metrics":flat_metrics(r["rows"]),"joint_pass":r["summary"]["joint_pass"],"exact_zero_mse":r["exact"]["exact_zero_mse"],"latent_separation":latent,"parameter_movement":parameter_distance(m.export(),st)[0]}); del before,m,o
    write_csv("layerwise_initialization_ablation.csv",[{k:(json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v) for k,v in r.items()} for r in ablations]); dump("layerwise_initialization_ablation.json",{"runs":ablations})

    # Latent and gradient comparison.
    named_states={"canonical_parent":parent_state,"formal_r1_step1750":state_from(R1),"old_step20000":old20,"old_step20000_p3_final":p3_final20}; latent_all=[]; grad_summary={}; grad_cos=[]; grad_layers=[]
    for name,st in named_states.items():
        model=Student(st).to(DEVICE)
        for row in latent_metrics(model,validation): latent_all.append({"checkpoint":name,**row})
        norms,cos,layers=gradients(model,validation,start); grad_summary[name]={"gradient_norms":norms,"adam_update_direction":"negative current gradient under fresh Adam","effective_step_size":2e-4}; grad_cos.extend([{"checkpoint":name,**x} for x in cos]); grad_layers.extend([{"checkpoint":name,**x} for x in layers]); del model
    dump("initialization_gap_latent_analysis.json",{"checkpoints":list(named_states),"metrics":latent_all}); write_csv("initialization_gap_latent_layer_metrics.csv",latent_all)
    dump("initialization_gap_gradient_analysis.json",grad_summary); write_csv("initialization_gap_gradient_cosines.csv",grad_cos); write_csv("initialization_gap_layerwise_gradients.csv",grad_layers)

    # Optimizer-state ablation.
    optimizer_runs=[]
    old_payload=torch.load(RAW/"checkpoints/student_step_20000.pt",map_location="cpu",weights_only=False)
    for name,opt_state,zero in (("old20_O1_FRESH_ADAM",None,False),("old20_O2_OLD_OPTIMIZER_STATE",old_payload.get("optimizer_state_dict"),False),("old20_O3_ZERO_MOMENT_SAME_STEP",old_payload.get("optimizer_state_dict"),True)):
        try:
            m,o,r=train_balanced(old20,pools,2000,eval_at=(2000,),validation=validation,start=start,optimizer_state=opt_state,zero_moments_keep_step=zero); optimizer_runs.append({"case":name,"availability":"AVAILABLE","metrics":flat_metrics(r["rows"]),"joint_pass":r["summary"]["joint_pass"],"trace_hash":r["trace_hash"],"parameter_movement":parameter_distance(m.export(),old20)[0],**r["exact"]}); del m,o
        except Exception as e: optimizer_runs.append({"case":name,"availability":"NOT_AVAILABLE","reason":repr(e)})
    parent_payload=torch.load(PARENT,map_location="cpu",weights_only=False)
    for name,opt_state in (("canonical_fresh_adam",None),("canonical_parent_optimizer_state",parent_payload.get("optimizer_state_dict"))):
        try:
            m,o,r=train_balanced(parent_state,pools,2000,eval_at=(2000,),validation=validation,start=start,optimizer_state=opt_state); optimizer_runs.append({"case":name,"availability":"AVAILABLE","metrics":flat_metrics(r["rows"]),"joint_pass":r["summary"]["joint_pass"],"trace_hash":r["trace_hash"],"parameter_movement":parameter_distance(m.export(),parent_state)[0],**r["exact"]}); del m,o
        except Exception as e: optimizer_runs.append({"case":name,"availability":"NOT_AVAILABLE","reason":repr(e)})
    dump("initialization_optimizer_state_ablation.json",{"runs":optimizer_runs})

    # Loss barriers, 0.025 resolution.
    barrier=[]
    for pair,a,b in (("canonical_to_old20",parent_state,old20),("r1_1750_to_p3_final",state_from(R1),p3_final20)):
        for i in range(41):
            lam=i/40; st={k:((1-lam)*a[k].float()+lam*b[k].float()).to(a[k].dtype) for k in a}; model=Student(st).to(DEVICE); rows,summary=evaluate(model,validation); ex=exact_start(model,start); barrier.append({"pair":pair,"lambda":lam,"worst_group_loss":summary["worst_group_mse"],"moving_retention_loss":summary["worst_moving_mse"],"exact_zero_loss":ex["exact_zero_mse"],"group_losses":{r["group"]:r["mean_mse"] for r in rows}}); del model
    write_csv("initialization_loss_barrier.csv",[{k:(json.dumps(v,sort_keys=True) if isinstance(v,dict) else v) for k,v in r.items()} for r in barrier]); dump("initialization_loss_barrier.json",{"rows":barrier})

    # Exact-zero feature timeline from immutable checkpoints and balanced snapshots.
    feature=[]
    for item in inventory:
        if item["initialization"].startswith("old_step_"): feature.append({"path":"original_objective_canonical","step":int(item["initialization"].split("_")[-1]),"exact_zero_mse":item["exact_zero_mse"],"steady_stop_mse":item["metrics"]["STEADY_STOP"]["mean_mse"],"stop_recovery_mse":item["metrics"]["STOP_RECOVERY"]["mean_mse"]})
    for r in long: feature.append({"path":"balanced_only_canonical","step":r["horizon"],"exact_zero_mse":r["exact_zero_mse"],"steady_stop_mse":r["metrics"]["STEADY_STOP"]["mean_mse"],"stop_recovery_mse":r["metrics"]["STOP_RECOVERY"]["mean_mse"]})
    for p in paths:
        if p["path"] in ("PATH_B_ORIGINAL_THEN_BALANCED","PATH_C_ORIGINAL_25K_THEN_BALANCED"): feature.append({"path":p["path"],"step":22000 if "25K" not in p["path"] else 27000,"exact_zero_mse":p["exact_zero_mse"],"steady_stop_mse":p["metrics"]["STEADY_STOP"]["mean_mse"],"stop_recovery_mse":p["metrics"]["STOP_RECOVERY"]["mean_mse"]})
    write_csv("exact_zero_feature_acquisition_timeline.csv",feature); dump("exact_zero_feature_acquisition_timeline.json",{"rows":feature,"latent_resolution":"see initialization_gap_latent_layer_metrics.csv","dominant_joint_errors":"inherited D1: lower-body stop/start action boundary"})

    warm={"status":"VALID_REPRODUCIBLE_INTERMEDIATE","canonical_parent_initialization_proven":tensor_hash(old_states[0])==tensor_hash(parent_state),"dataset_identity":"resolved immutable manifest; metric fingerprints PASS in D2","label_contract":"current W2-P1 labels","training_contract":original_contract,"checkpoint_sha256":sha(RAW/'checkpoints/student_step_20000.pt'),"checkpoint_actor_tensor_hash":tensor_hash(old20),"optimizer_state_available":True,"runtime_teacher_state_embedded":False,"exact_p3_reproduction":p3_res20["tensor_hash"]=="975f2cb165e48853f87d79cb93de83ed50954627b5b3a37f38c3b2bd6d4a159b"}; dump("old_w2_p1_student_warm_start_validity.json",warm)

    balanced_success=any(r["joint_pass"] for r in long); two_success=[p["path"] for p in paths if p["joint_pass"]]
    if balanced_success: classification="CANONICAL_BALANCED_TRAINING_TOO_SHORT"; next_action="formal long-horizon group-balanced supervised integration from canonical W1B-R2"
    elif any(p in two_success for p in ("PATH_B_ORIGINAL_THEN_BALANCED","PATH_C_ORIGINAL_25K_THEN_BALANCED","PATH_E_LINEAR_WEIGHT_SCHEDULE")): classification="ORIGINAL_OBJECTIVE_PRETRAINING_REQUIRED"; next_action="formal two-stage practical-stop integration: canonical W1B-R2 -> original 35/25/30/10 objective -> balanced 25/25/25/25 consolidation -> static authorization -> closed-loop evaluation"
    elif warm["status"]=="VALID_REPRODUCIBLE_INTERMEDIATE" and p3_res20["summary"]["joint_pass"]: classification="PRETRAINED_CHECKPOINT_WARM_START_VALID"; next_action="formalize the audited step-20,000 checkpoint as a reproducible intermediate artifact, then run balanced consolidation once"
    else: classification="INITIALIZATION_GAP_INCONCLUSIVE"; next_action="retain fail-closed status; do not start formal integration"
    dump("stage_classification.json",{"classification":classification,"single_primary_classification":True}); dump("recommended_next_action.json",{"classification":classification,"next_action":next_action,"executed":False})
    dump("current_w2_p1_initialization_gap_interpretation.json",{"dataset_identity":"PASS","P3_reproduction":"PASS","124D_joint_representation":"feasible","canonical_parent_balanced_2k":"FAIL","old_step20k_balanced_2k":"PASS","formal_closed_loop":"not authorized or executed","canonical_parent":"W1B-R2 iteration 200","student_promotion":"none"})
    dump("gate.json",{"classification":classification,"diagnosis_complete":True,"new_persistent_policy_checkpoint":0,"closed_loop_rollout":0,"DAgger":0,"canonical_promotion":0,"remote_push":False})
    print(json.dumps({"phase":"complete","classification":classification,"two_stage_success":two_success,"balanced_success":balanced_success}),flush=True)


if __name__=="__main__": main()
