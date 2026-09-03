"""In-memory-only fixed probes for W2-P1-D1.  No model is serialized."""
from __future__ import annotations

import csv
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve(); REPO = HERE.parents[4]; sys.path.insert(0, str(HERE.parent))
from train_w2_p1_student import MOVING_GROUPS, Student, load_datasets, sample, split_groups  # noqa: E402

SOURCE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_practical_stop_endpoint_acquisition"
RAW = SOURCE / "raw"
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_d1_static_representation_conflict_diagnosis"
GROUPS = ("STOP_RECOVERY", "STEADY_STOP", *MOVING_GROUPS, "START_RETENTION")


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def evaluate(model: Student, validation: dict[str, tuple[torch.Tensor,torch.Tensor,torch.Tensor]]) -> tuple[list[dict], dict]:
    rows=[]; model.eval()
    for group in GROUPS:
        obs,gait,target=validation[group]
        with torch.no_grad(): pred=model(obs,gait)
        mse=(pred-target).square().mean(1).cpu(); cosine=nn.functional.cosine_similarity(pred,target,dim=1).cpu()
        rows.append({"group":group,"mean_mse":float(mse.mean()),"cosine":float(cosine.mean()),"p95":float(torch.quantile(mse,.95)),"p99":float(torch.quantile(mse,.99)),"maximum":float(mse.max()),"gate_pass":bool(float(mse.mean())<=.001 and float(cosine.mean())>=.98)})
    return rows,{"all_group_gate_pass":all(r["gate_pass"] for r in rows),"moving_worst_mse":max(r["mean_mse"] for r in rows if r["group"] in MOVING_GROUPS)}


def draw(group: str, count: int, pools: dict[str,tuple[torch.Tensor,torch.Tensor,torch.Tensor]], nonzero: bool=False):
    pool=pools[group]
    eligible=torch.where(torch.linalg.vector_norm(pool[0][:,9:12],dim=1)>0)[0] if nonzero else torch.arange(len(pool[0]),device=pool[0].device)
    ids=eligible[torch.randint(len(eligible),(count,),device=pool[0].device)]
    return tuple(v[ids] for v in pool)


def objective(name: str, model: Student, pools: dict[str,tuple[torch.Tensor,torch.Tensor,torch.Tensor]]) -> torch.Tensor:
    def loss(group,count,nonzero=False):
        o,g,t=draw(group,count,pools,nonzero); return nn.functional.mse_loss(model(o,g),t)
    if name=="P1_START_ONLY": return loss("START_RETENTION",512)
    if name=="P2_START_PLUS_STEADY_STOP": return .5*loss("START_RETENTION",512)+.5*loss("STEADY_STOP",512)
    if name=="P3_ALL_GROUPS_BALANCED":
        moving=torch.stack([loss(g,64) for g in MOVING_GROUPS]).mean()
        return .25*(loss("STOP_RECOVERY",256)+loss("STEADY_STOP",256)+moving+loss("START_RETENTION",256))
    exclude=name=="P4_EXCLUDE_EXACT_ZERO_DIAGNOSTIC"
    moving=torch.stack([loss(g,64) for g in MOVING_GROUPS]).mean()
    return .35*loss("STOP_RECOVERY",256)+.25*loss("STEADY_STOP",256)+.30*moving+.10*loss("START_RETENTION",256,exclude)


def run_probe(name: str, steps: int, last_layer_only: bool, pools: dict, validation: dict, device: torch.device) -> tuple[list[dict],dict]:
    state=torch.load(RAW/"selected_w2_p1_student.pt",map_location="cpu",weights_only=False)["actor_state_dict"]
    model=Student(state).to(device); initial=torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    if last_layer_only:
        for p in model.parameters(): p.requires_grad_(False)
        for p in model.hidden[-1].parameters(): p.requires_grad_(True)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=2e-4)
    torch.manual_seed(20276050+sum(map(ord,name))); peak_grad=0.0; last_loss=0.0
    model.train()
    for _ in range(steps):
        loss=objective(name if name not in ("P5_LAST_LAYER_ONLY","P6_FULL_NETWORK") else "ORIGINAL",model,pools)
        opt.zero_grad(set_to_none=True); loss.backward(); grad=nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],10.0); opt.step()
        peak_grad=max(peak_grad,float(grad)); last_loss=float(loss)
    final=torch.cat([p.detach().flatten().cpu() for p in model.parameters()]); rows,summary=evaluate(model,validation)
    summary.update({"probe":name,"steps":steps,"last_layer_only":last_layer_only,"last_training_loss":last_loss,"peak_gradient_norm":peak_grad,"parameter_l2_movement":float(torch.linalg.vector_norm(final-initial)),"persistent_checkpoint_written":False})
    for row in rows: row.update({"probe":name,"steps":steps,"last_layer_only":last_layer_only,"parameter_l2_movement":summary["parameter_l2_movement"],"gradient_norm":peak_grad})
    del model,opt
    return rows,summary


def main() -> None:
    torch.manual_seed(20276021); device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets,groups=load_datasets(); splits=split_groups(datasets,groups)
    train_gen=torch.Generator().manual_seed(20276049); val_gen=torch.Generator().manual_seed(20276100)
    pools={g:sample(g,"train",50000,datasets,splits,train_gen,device) for g in GROUPS}
    validation={g:sample(g,"validation",5000,datasets,splits,val_gen,device) for g in GROUPS}
    del datasets,groups,splits
    specs=(("P1_START_ONLY",1000,False),("P2_START_PLUS_STEADY_STOP",2000,False),("P3_ALL_GROUPS_BALANCED",2000,False),("P4_EXCLUDE_EXACT_ZERO_DIAGNOSTIC",2000,False),("P5_LAST_LAYER_ONLY",5000,True),("P6_FULL_NETWORK",5000,False))
    all_rows=[]; summaries=[]
    for name,steps,last in specs:
        rows,summary=run_probe(name,steps,last,pools,validation,device); all_rows.extend(rows); summaries.append(summary); print(json.dumps(summary),flush=True)
    write_csv("static_representation_probe_training.csv",all_rows)
    feasible=[s["probe"] for s in summaries if s["all_group_gate_pass"]]
    payload={"fixed_split":"original episode-stratified training/validation split","probes":summaries,"rows":all_rows,"joint_feasible_probes":feasible,"persistent_checkpoint_writes":0,
        "representation_feasibility":"OPTIMIZATION_PATH_NOT_REPRESENTATION_LIMIT" if feasible else "NO_JOINT_STATIC_GATE_FOUND"}
    (OUT/"static_representation_probe_training.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")


if __name__=="__main__": main()
