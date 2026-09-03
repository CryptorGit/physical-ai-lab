"""Resolved-manifest W2-P1-R1 P3 reproduction and one persistent supervised run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve(); REPO = HERE.parents[4]; sys.path.insert(0, str(HERE.parent))
from train_w2_p1_student import MOVING_GROUPS, Student, load_datasets, sample, split_groups  # noqa: E402
from analyze_w2_p1_d1_static_conflict import heldout_start, predict  # noqa: E402

SOURCE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_practical_stop_endpoint_acquisition"
RAW = SOURCE / "raw"
RESOLUTION = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_r1_d2_dataset_provenance_reconciliation"
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_r1_group_balanced_stop_integration_resolved_manifest"
RUN_RAW = OUT / "raw"
PARENT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
PARENT_SHA = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"
P3_SOURCE = RAW / "selected_w2_p1_student.pt"
P3_SOURCE_SHA = "dcafa8d18699f7fb5860ba392d0c8cf3ee56d175eefe0793d45830cee1b44130"
GROUPS = ("STOP_RECOVERY", "STEADY_STOP", *MOVING_GROUPS, "START_RETENTION")
SAVE_STEPS = (0, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_identity() -> dict:
    resolved = json.loads((RESOLUTION / "w2_p1_dataset_hashes_resolved_v2.json").read_text())
    if resolved["status"] != "IMMUTABLE_RESOLVED_SOURCE_OF_TRUTH": raise RuntimeError("RESOLVED_MANIFEST_STATUS_FAIL")
    actual = {key: sha(REPO / key) for key in resolved["hashes"]}
    split_sha = sha(SOURCE / "w2_p1_dataset_split.json")
    gate = json.loads((RESOLUTION / "dataset_provenance_gate.json").read_text())
    passed = actual == resolved["hashes"] and split_sha == resolved["split_sha256"] and gate["next_stage_group_balanced_training_authorized"]
    payload = {"resolved_manifest": str((RESOLUTION / "w2_p1_dataset_hashes_resolved_v2.json").relative_to(REPO)).replace("\\", "/"), "status": resolved["status"], "expected_hashes": resolved["hashes"], "actual_hashes": actual, "semantic_hashes": resolved["semantic_hashes"], "split_expected": resolved["split_sha256"], "split_actual": split_sha, "ordered_sample_identity": "adbf4fd157542ebe602dc75980013779813c6c51179140c87b99e95b9b0e056d", "identity_gate_pass": passed, "original_stale_manifest_used_for_authorization": False}
    if not passed: raise RuntimeError("RESOLVED_DATASET_IDENTITY_FAIL")
    return payload


def make_data(device: torch.device):
    datasets, groups = load_datasets(); splits = split_groups(datasets, groups)
    pool_gen = torch.Generator().manual_seed(20276049); val_gen = torch.Generator().manual_seed(20276100)
    pools = {g: sample(g, "train", 50000, datasets, splits, pool_gen, device) for g in GROUPS}
    validation = {g: sample(g, "validation", 5000, datasets, splits, val_gen, device) for g in GROUPS}
    return datasets, splits, pools, validation


def evaluate_pool(model: Student, values: dict) -> tuple[list[dict], dict]:
    rows=[]; model.eval()
    with torch.inference_mode():
        for group in GROUPS:
            obs, gait, target = values[group]; pred = model(obs, gait)
            mse = (pred-target).square().mean(1); cosine = nn.functional.cosine_similarity(pred, target, dim=1)
            rows.append({"group": group, "mean_mse": float(mse.mean()), "cosine": float(cosine.mean()), "p95": float(torch.quantile(mse,.95)), "p99": float(torch.quantile(mse,.99)), "maximum": float(mse.max()), "gate_pass": bool(float(mse.mean()) <= .001 and float(cosine.mean()) >= .98)})
    return rows, {"all_group_gate_pass": all(r["gate_pass"] for r in rows), "moving_worst_mse": max(r["mean_mse"] for r in rows if r["group"] in MOVING_GROUPS)}


def draw(group: str, count: int, pools: dict):
    values = pools[group]; ids = torch.randint(len(values[0]), (count,), device=values[0].device)
    return tuple(value[ids] for value in values)


def balanced_loss(model: Student, pools: dict):
    def loss(group, count):
        o,g,t = draw(group,count,pools); return nn.functional.mse_loss(model(o,g),t)
    # Preserve the D1 P3 RNG consumption order exactly: moving subgroups are
    # drawn before the three top-level groups in the return expression.
    moving=torch.stack([loss(g,64) for g in MOVING_GROUPS]).mean(); stop=loss("STOP_RECOVERY",256); steady=loss("STEADY_STOP",256); start=loss("START_RETENTION",256)
    return .25*(stop+steady+moving+start), {"stop_recovery_loss":float(stop),"steady_stop_loss":float(steady),"moving_retention_loss":float(moving),"start_retention_loss":float(start)}


def init_model(device: torch.device, p3_source: bool=False) -> Student:
    source, expected = (P3_SOURCE, P3_SOURCE_SHA) if p3_source else (PARENT, PARENT_SHA)
    if sha(source) != expected: raise RuntimeError("INITIALIZATION_IDENTITY_FAIL")
    payload=torch.load(source,map_location="cpu",weights_only=False)
    return Student(payload["actor_state_dict"]).to(device)


def run_once(pools: dict, validation: dict, device: torch.device, save: bool=False, p3_source: bool=False):
    model=init_model(device,p3_source); optimizer=torch.optim.Adam(model.parameters(),lr=2e-4)
    torch.manual_seed(20277717); random.seed(20277717); trace=[]; peak=0.0
    checkpoints=[]
    if save:
        RUN_RAW.mkdir(parents=True,exist_ok=True); checkpoints.append(save_checkpoint(model,optimizer,0,{}))
    model.train()
    for step in range(1,2001):
        loss,parts=balanced_loss(model,pools); optimizer.zero_grad(set_to_none=True); loss.backward(); grad=nn.utils.clip_grad_norm_(model.parameters(),10.0); optimizer.step(); peak=max(peak,float(grad))
        if step % 10 == 0: trace.append({"step":step,"loss":float(loss),"gradient_norm":float(grad),**parts})
        if save and step in SAVE_STEPS: checkpoints.append(save_checkpoint(model,optimizer,step,{}))
    rows,summary=evaluate_pool(model,validation); summary.update({"peak_gradient_norm":peak,"last_training_loss":trace[-1]["loss"],"tensor_hash":tensor_hash(model.export()),"trace_hash":hashlib.sha256(json.dumps(trace,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"persistent":save})
    return model,rows,summary,trace,checkpoints


def save_checkpoint(model,optimizer,step,metrics):
    path=RUN_RAW/f"student_step_{step}.pt"
    torch.save({"step":step,"actor_state_dict":model.export(),"optimizer_state_dict":optimizer.state_dict(),"static_metrics":metrics,"architecture":[124,256,128,128,37],"std_frozen":True,"resolved_manifest":str((RESOLUTION/'w2_p1_dataset_hashes_resolved_v2.json').relative_to(REPO)).replace('\\','/')},path)
    return path


def p3_same(output: Path):
    identity=verify_identity(); device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); _,_,pools,validation=make_data(device)
    runs=[]
    for index in range(2):
        model,rows,summary,trace,_=run_once(pools,validation,device,False,True); runs.append({"index":index,"rows":rows,"summary":summary,"initialization":"D1 selected step-20000 student, as executed by the immutable D1 P3 source"}); del model
    parity=runs[0]["summary"]["tensor_hash"]==runs[1]["summary"]["tensor_hash"] and runs[0]["summary"]["trace_hash"]==runs[1]["summary"]["trace_hash"] and runs[0]["rows"]==runs[1]["rows"]
    dump(output,{"identity":identity,"runs":runs,"same_process_parity":parity})
    print(json.dumps({"same_process_parity":parity,"tensor_hash":runs[0]["summary"]["tensor_hash"],"gate":runs[0]["summary"]["all_group_gate_pass"]}))


def p3_fresh(output: Path):
    identity=verify_identity(); device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); _,_,pools,validation=make_data(device)
    model,rows,summary,trace,_=run_once(pools,validation,device,False,True); dump(output,{"identity":identity,"rows":rows,"summary":summary,"initialization":"D1 selected step-20000 student, as executed by the immutable D1 P3 source"}); print(json.dumps(summary))


def formal_train():
    identity=verify_identity(); OUT.mkdir(parents=True,exist_ok=True); device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets,splits,pools,validation=make_data(device); model,_,_,trace,checkpoint_paths=run_once(pools,validation,device,True)
    # Evaluate every saved checkpoint on a fixed held-out pool.
    held_gen=torch.Generator().manual_seed(20276023); held={g:sample(g,"held_out",10000,datasets,splits,held_gen,device) for g in GROUPS}
    timeline=[]; candidates=[]
    parent_state=torch.load(PARENT,map_location="cpu",weights_only=False)["actor_state_dict"]
    parent_vector=torch.cat([v.flatten().float() for k,v in sorted(parent_state.items()) if not k.startswith("distribution.")])
    for path in checkpoint_paths:
        payload=torch.load(path,map_location="cpu",weights_only=False); candidate=Student(payload["actor_state_dict"]).to(device); rows,summary=evaluate_pool(candidate,held)
        vector=torch.cat([v.flatten().float() for k,v in sorted(candidate.export().items()) if not k.startswith("distribution.")]); movement=float(torch.linalg.vector_norm(vector.cpu()-parent_vector))
        for row in rows: timeline.append({"step":payload["step"],**row,"parameter_l2_movement":movement})
        aggregate=sum(r["mean_mse"] for r in rows)/len(rows); rank=(not summary["all_group_gate_pass"],next(r["mean_mse"] for r in rows if r["group"]=="START_RETENTION"),next(r["mean_mse"] for r in rows if r["group"]=="STOP_RECOVERY"),next(r["mean_mse"] for r in rows if r["group"]=="STEADY_STOP"),summary["moving_worst_mse"],aggregate,movement,payload["step"])
        candidates.append((rank,path,rows,summary,movement))
    fields=list(timeline[0]);
    with (OUT/"group_balanced_static_checkpoint_timeline.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(timeline)
    dump(OUT/"group_balanced_static_checkpoint_timeline.json",{"rows":timeline})
    rank,path,rows,summary,movement=min(candidates,key=lambda x:x[0])
    if not summary["all_group_gate_pass"]: raise RuntimeError("EXP013_W2_P1_R1_STATIC_REPRODUCTION_FAIL")
    selected=torch.load(path,map_location="cpu",weights_only=False); selected_path=RUN_RAW/"selected_student.pt"; torch.save({**selected,"held_out_rows":rows,"selection_rank":rank},selected_path)
    # Exact-zero diagnostics retain every sample.
    start=heldout_start(datasets,splits); selected_model=Student(selected["actor_state_dict"]).to(device); pred=predict(selected_model,start["observation"],start["gait_cmd"],device); mse=(pred-start["target_action"]).square().mean(1); zero=torch.linalg.vector_norm(start["physical_command"],dim=1)==0
    exact={"sample_count":len(mse),"exact_zero_count":int(zero.sum()),"exact_zero_mse":float(mse[zero].mean()),"nonzero_mse":float(mse[~zero].mean()),"all_samples_retained":True}
    dump(OUT/"static_heldout_results.json",{"rows":rows,"all_group_gate_pass":summary["all_group_gate_pass"]}); dump(OUT/"exact_zero_boundary_results.json",exact)
    dump(OUT/"selected_checkpoint.json",{"step":selected["step"],"path":str(selected_path.relative_to(REPO)).replace('\\','/'),"sha256":sha(selected_path),"tensor_hash":tensor_hash(selected["actor_state_dict"]),"selection_rank":list(rank),"static_gate":"PASS"})
    dump(OUT/"checkpoint_manifest.json",{"checkpoints":[{"step":torch.load(p,map_location='cpu',weights_only=False)["step"],"path":str(p.relative_to(REPO)).replace('\\','/'),"sha256":sha(p)} for p in checkpoint_paths],"selected":str(selected_path.relative_to(REPO)).replace('\\','/')})
    with (OUT/"training_curves.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(trace[0]));w.writeheader();w.writerows(trace)
    dump(OUT/"resolved_dataset_identity_audit.json",identity); dump(OUT/"resolved_dataset_hashes.json",{"byte_hashes":identity["actual_hashes"],"semantic_hashes":identity["semantic_hashes"],"split_sha256":identity["split_actual"]}); dump(OUT/"dataset_identity_gate.json",{"pass":True,"dataset_bytes_changed":0,"label_bytes_changed":0,"split_changed":0})
    print(json.dumps({"selected_step":selected["step"],"selected_sha":sha(selected_path),"static_gate":True,"exact":exact}))


def parity(output: Path):
    identity=verify_identity(); selected=json.loads((OUT/"selected_checkpoint.json").read_text()); path=REPO/selected["path"]; payload=torch.load(path,map_location="cpu",weights_only=False); device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); datasets,groups=load_datasets();splits=split_groups(datasets,groups);gen=torch.Generator().manual_seed(20276023);held={g:sample(g,"held_out",10000,datasets,splits,gen,device) for g in GROUPS}; model=Student(payload["actor_state_dict"]).to(device);rows,summary=evaluate_pool(model,held); dump(output,{"identity":identity,"parameter_hash":tensor_hash(payload["actor_state_dict"]),"checkpoint_sha256":sha(path),"rows":rows,"summary":summary})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("p3-same","p3-fresh","train","parity"));parser.add_argument("--output",type=Path);args=parser.parse_args()
    if args.mode=="p3-same":p3_same(args.output)
    elif args.mode=="p3-fresh":p3_fresh(args.output)
    elif args.mode=="train":formal_train()
    else:parity(args.output)

if __name__=="__main__": main()
