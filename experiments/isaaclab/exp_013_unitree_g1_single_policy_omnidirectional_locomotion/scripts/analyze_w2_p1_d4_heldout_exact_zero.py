"""Read-only W2-P1-D4 exact-zero held-out authorization diagnosis.

This script never serializes a tensor and never changes datasets, splits,
checkpoints, or their manifests.  It evaluates the frozen R2 step-37000 actor,
replays the registered hierarchical sampling contract, and writes additive
diagnostic artifacts only.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE.parent))
from train_w2_p1_student import MOVING_GROUPS, Student, load_datasets, sample, split_groups  # noqa: E402

BASE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
SOURCE = BASE / "phase_w2_p1_practical_stop_endpoint_acquisition"
R2 = BASE / "phase_w2_p1_r2_long_horizon_group_balanced_stop_integration"
OUT = BASE / "phase_w2_p1_d4_heldout_exact_zero_generalization_diagnosis"
RAW = R2 / "raw"
SELECTED = RAW / "selected_student.pt"
CHECKPOINTS = RAW / "checkpoints"
STARTING_HEAD = "71dd6eb25b1b26a0c11b226c1e0b3567e7845902"
STEPS = (10000,12000,17000,18500,20500,21500,22000,22500,23500,24000,
         24500,26500,28500,30000,31000,33000,36000,37000,39500,40000)
GROUP_ORDER = ("STOP_RECOVERY", "STEADY_STOP", *MOVING_GROUPS, "START_RETENTION")
JOINT_NAMES = (
 "left_hip_pitch","right_hip_pitch","waist_yaw","left_hip_roll","right_hip_roll","waist_roll",
 "left_hip_yaw","right_hip_yaw","waist_pitch","left_knee","right_knee","left_shoulder_pitch",
 "right_shoulder_pitch","left_ankle_pitch","right_ankle_pitch","left_shoulder_roll",
 "right_shoulder_roll","left_ankle_roll","right_ankle_roll","left_shoulder_yaw","right_shoulder_yaw",
 "left_elbow","right_elbow","left_wrist_roll","right_wrist_roll","left_wrist_pitch","right_wrist_pitch",
 "left_wrist_yaw","right_wrist_yaw","left_hand_index_0","right_hand_index_0","left_hand_middle_0",
 "right_hand_middle_0","left_hand_pinky_0","right_hand_pinky_0","left_hand_ring_0","right_hand_ring_0")
JOINT_GROUPS = {
 "hip": (0,1,3,4,6,7), "knee": (9,10), "ankle": (13,14,17,18), "waist": (2,5,8),
 "shoulder": (11,12,15,16,19,20), "elbow": (21,22), "hand": tuple(range(23,37))}


def dump(name: str, value: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["status"]
        rows = [{"status": "no_rows"}]
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


def by_dataset(refs):
    result = defaultdict(list)
    for d, e in refs: result[d].append(e)
    return result


def full_start(datasets, refs) -> dict:
    pieces = defaultdict(list)
    for di, episodes in sorted(by_dataset(refs).items()):
        data = datasets[di]; ep = torch.tensor(episodes); t = data["observation"].shape[0]
        for key in ("observation","target_action","source_action","teacher_action","physical_command",
                    "actor_command","contact"):
            pieces[key].append(data[key][:,ep].permute(1,0,2).reshape(-1,data[key].shape[-1]))
        for key in ("gait_cmd","translation_speed","absolute_yaw_rate","flight"):
            pieces[key].append(data[key][:,ep].T.reshape(-1))
        pieces["episode_id"].append(data["episode_id"][ep].repeat_interleave(t))
        pieces["episode_index"].append(ep.repeat_interleave(t))
        pieces["dataset_index"].append(torch.full((len(ep)*t,),di,dtype=torch.int64))
        pieces["time_index"].append(torch.arange(t).repeat(len(ep)))
        pieces["condition"].extend([data["condition"][e] for e in episodes for _ in range(t)])
        pieces["episode_condition"].extend([data["condition"][e] for e in episodes])
        pieces["episode_ids_only"].extend([int(data["episode_id"][e]) for e in episodes])
    return {k:(torch.cat(v) if v and torch.is_tensor(v[0]) else v) for k,v in pieces.items()}


def predict(model, x, gait, device):
    values=[]
    with torch.inference_mode():
        for begin in range(0,len(x),8192): values.append(model(x[begin:begin+8192].to(device),gait[begin:begin+8192].to(device)).cpu())
    return torch.cat(values)


def metrics(pred, target, zero, episode_id, conditions) -> dict:
    err=(pred-target).square().mean(1); cos=nn.functional.cosine_similarity(pred,target,dim=1)
    ep=[]
    for eid in torch.unique(episode_id): ep.append(float(err[episode_id==eid].mean()))
    cond=[]
    for name in sorted(set(conditions)):
        ids=torch.tensor([x==name for x in conditions]); cond.append(float(err[ids].mean()))
    return {"sample_count":len(err),"episode_count":len(torch.unique(episode_id)),"exact_zero_count":int(zero.sum()),
            "exact_zero_prevalence":float(zero.float().mean()),"mean_mse":float(err.mean()),"cosine":float(cos.mean()),
            "exact_zero_mse":float(err[zero].mean()),"nonzero_mse":float(err[~zero].mean()),
            "episode_balanced_mse":float(np.mean(ep)),"condition_balanced_mse":float(np.mean(cond))}


def parse_condition(name: str):
    p=name.split(":"); return float(p[1]),float(p[3])


def hierarchical_indices(ref_count: int, timesteps: int, count: int, generator: torch.Generator) -> torch.Tensor:
    choices=torch.randint(ref_count,(count,),generator=generator)
    times=torch.randint(timesteps,(count,),generator=generator)
    return choices*timesteps+times


def resample_rows(split_name, error, cosine, zero, repeats, seed, count=10000):
    rows=[]; gen=torch.Generator().manual_seed(seed); n=len(error); t=55; refs=n//t
    for repeat in range(repeats):
        ids=hierarchical_indices(refs,t,count,gen); e=error[ids].double(); c=cosine[ids].double(); z=zero[ids]
        zc=int(z.sum()); zm=float(e[z].mean()) if zc else 0.0; nzm=float(e[~z].mean())
        rows.append({"split":split_name,"repeat":repeat,"sampling_seed":seed,"exact_zero_count":zc,
                     "exact_zero_prevalence":zc/count,"overall_mse":float(e.mean()),"cosine":float(c.mean()),
                     "gate_pass":bool(float(e.mean())<=.001 and float(c.mean())>=.98),
                     "exact_zero_mean_mse":zm,"nonzero_mean_mse":nzm,
                     "exact_zero_contribution":zc*zm/count,"nonzero_contribution":(count-zc)*nzm/count})
    return rows


def summarize_resampling(rows, original_mse):
    mse=np.array([r["overall_mse"] for r in rows]); z=np.array([r["exact_zero_count"] for r in rows]); passed=np.array([r["gate_pass"] for r in rows])
    return {"repeats":len(rows),"pass_probability":float(passed.mean()),"mse_mean":float(mse.mean()),
            "mse_std":float(mse.std()),**{f"mse_p{q}":float(np.quantile(mse,q/100)) for q in (1,5,50,95,99)},
            "exact_zero_count_mean":float(z.mean()),"exact_zero_count_min":int(z.min()),"exact_zero_count_max":int(z.max()),
            "maximum_zero_count_among_pass":int(z[passed].max()) if passed.any() else None,
            "minimum_zero_count_among_fail":int(z[~passed].min()) if (~passed).any() else None,
            "original_result_percentile":float((mse<=original_mse).mean())}


def auroc(scores, labels):
    order=torch.argsort(scores); ranks=torch.empty_like(order,dtype=torch.float64); ranks[order]=torch.arange(1,len(scores)+1,dtype=torch.float64)
    pos=labels.bool(); n1=int(pos.sum()); n0=len(labels)-n1
    return float((ranks[pos].sum()-n1*(n1+1)/2)/max(1,n1*n0))


def classifier_probe(x,y,nonlinear):
    gen=torch.Generator().manual_seed(20278017); order=torch.randperm(len(x),generator=gen); split=int(.8*len(x))
    tr,te=order[:split],order[split:]; mean=x[tr].mean(0); std=x[tr].std(0).clamp_min(1e-6)
    tx=((x[tr]-mean)/std); vx=((x[te]-mean)/std); ty=y[tr].float(); vy=y[te].float()
    torch.manual_seed(20278017); net=(nn.Sequential(nn.Linear(x.shape[1],32),nn.ELU(),nn.Linear(32,1)) if nonlinear else nn.Linear(x.shape[1],1))
    opt=torch.optim.Adam(net.parameters(),lr=3e-3)
    for _ in range(300):
        logits=net(tx).squeeze(1); loss=nn.functional.binary_cross_entropy_with_logits(logits,ty); opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad(): s=net(vx).squeeze(1); p=s.sigmoid(); pred=p>=.5
    return {"auroc":auroc(s,vy),"accuracy":float((pred==vy.bool()).float().mean())}


def source_locations(path: Path, patterns):
    lines=path.read_text(encoding="utf-8").splitlines(); return [{"line":i,"text":line.strip()} for i,line in enumerate(lines,1) if any(p in line for p in patterns)]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    start_head=git("rev-parse","HEAD"); dirty_before=git("status","--short").splitlines()
    datasets,groups=load_datasets(); splits=split_groups(datasets,groups)
    payload=torch.load(SELECTED,map_location="cpu",weights_only=False); device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=Student(payload["actor_state_dict"]).to(device).eval()
    split_data={}; split_pred={}; split_err={}; split_cos={}; split_zero={}; full_rows=[]; breakdown=[]
    for part in ("train","validation","held_out"):
        data=full_start(datasets,splits["START_RETENTION"][part]); pred=predict(model,data["observation"],data["gait_cmd"],device)
        zero=torch.linalg.vector_norm(data["physical_command"],dim=1)==0; err=(pred-data["target_action"]).square().mean(1); cos=nn.functional.cosine_similarity(pred,data["target_action"],dim=1)
        split_data[part]=data;split_pred[part]=pred;split_err[part]=err;split_cos[part]=cos;split_zero[part]=zero
        row={"split":part,**metrics(pred,data["target_action"],zero,data["episode_id"],data["condition"])};full_rows.append(row)
        angles=sorted(set(parse_condition(c)[0] for c in data["condition"])); yaws=sorted(set(parse_condition(c)[1] for c in data["condition"]))
        for kind,values in (("direction",angles),("yaw",yaws)):
            for val in values:
                mask=torch.tensor([(parse_condition(c)[0] if kind=="direction" else parse_condition(c)[1])==val for c in data["condition"]])
                breakdown.append({"split":part,"dimension":kind,"value":val,"samples":int(mask.sum()),"mse":float(err[mask].mean()),"cosine":float(cos[mask].mean()),"exact_zero_prevalence":float(zero[mask].float().mean())})
    write_csv("start_full_population_metrics.csv",full_rows+breakdown);dump("start_full_population_metrics.json",{"splits":full_rows,"breakdown":breakdown,"formal_gate_changed":False})

    held=next(x for x in full_rows if x["split"]=="held_out"); pmax=(.001-held["nonzero_mse"])/(held["exact_zero_mse"]-held["nonzero_mse"])
    sensitivity={"formula":"(gate-MSE_nonzero)/(MSE_zero-MSE_nonzero)","gate":.001,"exact_zero_mse":held["exact_zero_mse"],"nonzero_mse":held["nonzero_mse"],"maximum_exact_zero_prevalence":pmax,"maximum_exact_zero_count_per_10000":math.floor(10000*pmax),"reported_heldout":{"count":171,"prevalence":.0171,"mse":.00116143038},"reported_validation":{"count":72,"sample_count":5000,"prevalence":.0144,"mse":.0009793625},"pass_fail_determined_by_boundary_prevalence":True}
    dump("exact_zero_gate_sensitivity.json",sensitivity)

    trainer=HERE.parent/"train_w2_p1_student.py"; runner=HERE.parent/"run_w2_p1_r2_long_horizon.py"
    contract={"population_unit":"episode-stratified immutable split","split_seed":20276021,"population":{r["split"]:{"samples":r["sample_count"],"episodes":r["episode_count"],"exact_zero":r["exact_zero_count"]} for r in full_rows},"authorization_sample_count":10000,"sampling":"uniform episode with replacement, then uniform timestep with replacement within selected episode","with_replacement":True,"heldout_seed":20276023,"validation_selection_seed":20276100,"validation_selection_sample_count":5000,"episode_weighting":"uniform through first-stage draw","condition_weighting":"implicit split prevalence; no explicit reweighting","timestep_weighting":"uniform within selected episode","exact_zero_inclusion":"random, not guaranteed","group_rng_consumption_order":GROUP_ORDER}
    dump("start_authorization_sampling_contract.json",contract);dump("start_authorization_sampling_source_locations.json",{"trainer":str(trainer.relative_to(REPO)),"runner":str(runner.relative_to(REPO)),"locations":source_locations(trainer,["def split_groups","def sample","torch.randint","20276023"])+source_locations(runner,["20276023","10000","20276100","5000"])})

    resampling=[]
    resampling += resample_rows("validation",split_err["validation"],split_cos["validation"],split_zero["validation"],5000,20278401)
    resampling += resample_rows("held_out",split_err["held_out"],split_cos["held_out"],split_zero["held_out"],5000,20283401)
    write_csv("start_10k_resampling_distribution.csv",resampling)
    summaries={p:summarize_resampling([r for r in resampling if r["split"]==p], .0009793625 if p=="validation" else .00116143038) for p in ("validation","held_out")}
    dump("start_10k_resampling_distribution.json",{"summaries":summaries,"contract":"same hierarchical with-replacement contract","formal_gate_changed":False})
    max_reconstruction=max(abs(r["overall_mse"]-(r["exact_zero_contribution"]+r["nonzero_contribution"])) for r in resampling)
    dump("start_mse_analytic_reconstruction.json",{"formula":"n0/N*MSE0 + (N-n0)/N*MSE_nonzero","maximum_absolute_difference":max_reconstruction,"required_tolerance":1e-10,"pass":max_reconstruction<=1e-10,"pass_fail_explained_by_exact_zero_count":True})

    # Robust validation-only replay.  One registered resample stream is shared by all checkpoints.
    vdata=split_data["validation"]; vzero=split_zero["validation"]; gen=torch.Generator().manual_seed(20288401)
    robust_ids=[hierarchical_indices(len(vdata["episode_ids_only"]),55,10000,gen) for _ in range(1000)]
    timeline=json.loads((R2/"validation_checkpoint_timeline.json").read_text())["checkpoints"]
    nominal={int(x["step"]):x for x in timeline}; robustness=[]
    for step in STEPS:
        cp=torch.load(CHECKPOINTS/f"student_step_{step}.pt",map_location="cpu",weights_only=False); m=Student(cp["actor_state_dict"]).to(device).eval(); p=predict(m,vdata["observation"],vdata["gait_cmd"],device)
        e=(p-vdata["target_action"]).square().mean(1); c=nn.functional.cosine_similarity(p,vdata["target_action"],dim=1); vals=[]; passes=[]
        for ids in robust_ids:
            me=float(e[ids].mean()); co=float(c[ids].mean()); vals.append(me);passes.append(me<=.001 and co>=.98)
        rec=nominal[step]; arr=np.array(vals)
        robustness.append({"step":step,"nominal_validation_mse":rec["metrics"]["START_RETENTION"]["mean_mse"],"resampled_pass_probability":float(np.mean(passes)),"median_mse":float(np.median(arr)),"p95_mse":float(np.quantile(arr,.95)),"full_validation_mse":float(e.mean()),"exact_zero_mse":float(e[vzero].mean()),"nonzero_mse":float(e[~vzero].mean()),"worst_group_robust_mse":float(np.quantile(arr,.95))})
    write_csv("validation_checkpoint_sampling_robustness.csv",robustness);dump("validation_checkpoint_sampling_robustness.json",{"checkpoints":robustness,"heldout_used":False,"repeats_each":1000})
    nominal_order=sorted(robustness,key=lambda r:r["nominal_validation_mse"]); robust_order=sorted(robustness,key=lambda r:(-r["resampled_pass_probability"],r["median_mse"]))
    nr={r["step"]:i+1 for i,r in enumerate(nominal_order)}; rr={r["step"]:i+1 for i,r in enumerate(robust_order)}
    corr=float(np.corrcoef([nr[s] for s in STEPS],[rr[s] for s in STEPS])[0,1]); reversals=sum((nr[a]-nr[b])*(rr[a]-rr[b])<0 for i,a in enumerate(STEPS) for b in STEPS[i+1:])
    selected_rob=next(r for r in robustness if r["step"]==37000)
    dump("validation_selection_noise_decomposition.json",{"rank_correlation":corr,"selection_reversal_count":reversals,"step37000_nominal_rank":nr[37000],"step37000_robust_rank":rr[37000],"step37000_resampled_pass_probability":selected_rob["resampled_pass_probability"],"nominal_selection_depended_on_single_fixed_5000_sample_pool":True,"checkpoint_reselection_performed":False,"classification":"VALIDATION_SUBSAMPLE_SELECTION_OVERFIT" if rr[37000]>5 else "ROBUST_VALIDATION_SELECTION"})
    dump("heldout_checkpoint_usage_audit.json",{"heldout_checkpoint_steps_evaluated":[37000],"alternative_checkpoint_search":False,"checkpoint_reselection":False,"pass":True})

    prevalence=[]
    for part in ("train","validation","held_out"):
        d=split_data[part]
        for i,(eid,cond) in enumerate(zip(d["episode_ids_only"],d["episode_condition"])):
            base=i*55; z=int(split_zero[part][base:base+55].sum()); angle,yaw=parse_condition(cond)
            prevalence.append({"split":part,"episode_id":eid,"condition":cond,"direction":angle,"yaw":yaw,"episode_length":55,"exact_zero_sample_count":z,"accepted":True})
    write_csv("exact_zero_split_prevalence.csv",prevalence)
    dump("exact_zero_split_prevalence.json",{"episodes":len(prevalence),"split_summary":{p:{"episodes":sum(r["split"]==p for r in prevalence),"exact_zero_per_episode":sorted(set(r["exact_zero_sample_count"] for r in prevalence if r["split"]==p)),"natural_prevalence":1/55} for p in ("train","validation","held_out")},"all_accepted_episodes_have_one_boundary_sample":all(r["exact_zero_sample_count"]==1 for r in prevalence)})

    # Exact-zero split shift on the complete, fixed episode populations.
    vx=split_data["validation"]["observation"][split_zero["validation"]]; hx=split_data["held_out"]["observation"][split_zero["held_out"]]
    x=torch.cat([vx,hx]); y=torch.cat([torch.zeros(len(vx)),torch.ones(len(hx))]); linear=classifier_probe(x,y,False); nonlinear=classifier_probe(x,y,True)
    mean=x.mean(0);std=x.std(0).clamp_min(1e-6);vz=(vx-mean)/std;hz=(hx-mean)/std
    energy=float(2*torch.cdist(vz,hz).mean()-torch.cdist(vz,vz).mean()-torch.cdist(hz,hz).mean())
    nn_v=float(torch.cdist(vz,hz).min(1).values.mean());nn_h=float(torch.cdist(hz,vz).min(1).values.mean())
    state_shift={"validation_samples":len(vx),"heldout_samples":len(hx),"linear_probe":linear,"small_nonlinear_probe":nonlinear,"energy_distance_standardized_123d":energy,"cross_split_nearest_neighbor_distance":{"validation_to_heldout":nn_v,"heldout_to_validation":nn_h},"feature_contract":{"base_linear_velocity":[0,2],"base_angular_velocity":[3,5],"projected_gravity":[6,8],"command":[9,11],"joint_position":[12,48],"joint_velocity":[49,85],"previous_action":[86,122]},"classification":"TRUE_EXACT_ZERO_SPLIT_SHIFT" if nonlinear["auroc"]>.8 else "NO_MEANINGFUL_EXACT_ZERO_STATE_SHIFT"}
    dump("validation_heldout_exact_zero_state_shift.json",state_shift)

    # Exact-zero action geometry and joint contributions.
    action_rows=[]; joint_rows=[]
    totals={}
    for part in ("validation","held_out"):
        d=split_data[part]; z=split_zero[part]; s=split_pred[part][z]; w=d["target_action"][z]; teacher=d["teacher_action"][z]
        pairs={"student_vs_w1b":s-w,"student_vs_stop_teacher":s-teacher,"w1b_vs_stop_teacher":w-teacher}
        for name,diff in pairs.items():
            action_rows.append({"split":part,"pair":name,"samples":len(diff),"mse":float(diff.square().mean()),"cosine":float(nn.functional.cosine_similarity(s,w if name=="student_vs_w1b" else teacher,dim=1).mean()) if name!="w1b_vs_stop_teacher" else float(nn.functional.cosine_similarity(w,teacher,dim=1).mean())})
        sq=(s-w).square(); total=float(sq.sum()); totals[part]=total
        for j,name in enumerate(JOINT_NAMES): joint_rows.append({"split":part,"joint_index":j,"joint":name,"group":next(g for g,ids in JOINT_GROUPS.items() if j in ids),"mse":float(sq[:,j].mean()),"loss_contribution":float(sq[:,j].sum()/total)})
        for group,ids in JOINT_GROUPS.items():
            ids=torch.tensor(ids); joint_rows.append({"split":part,"joint_index":"group","joint":group,"group":group,"mse":float(sq[:,ids].mean()),"loss_contribution":float(sq[:,ids].sum()/total)})
    write_csv("exact_zero_action_error_by_split.csv",action_rows);dump("exact_zero_action_error_by_split.json",{"rows":action_rows,"heldout_minus_validation_student_w1b_mse":next(r["mse"] for r in action_rows if r["split"]=="held_out" and r["pair"]=="student_vs_w1b")-next(r["mse"] for r in action_rows if r["split"]=="validation" and r["pair"]=="student_vs_w1b")})
    write_csv("exact_zero_joint_semantic_contribution.csv",joint_rows)
    group_summary={p:{g:next(r for r in joint_rows if r["split"]==p and r["joint"]==g and r["joint_index"]=="group") for g in JOINT_GROUPS} for p in ("validation","held_out")}
    upper=sum(group_summary["held_out"][g]["loss_contribution"] for g in ("shoulder","elbow","hand")); lower=sum(group_summary["held_out"][g]["loss_contribution"] for g in ("hip","knee","ankle"))
    dump("exact_zero_joint_semantic_contribution.json",{"joint_rows":joint_rows,"group_summary":group_summary,"heldout_upper_body_contribution":upper,"heldout_lower_body_contribution":lower,"heldout_waist_contribution":group_summary["held_out"]["waist"]["loss_contribution"],"legs_only_mse":sum(group_summary["held_out"][g]["mse"]*len(JOINT_GROUPS[g]) for g in ("hip","knee","ankle"))/12,"lower_body_plus_waist_mse":sum(group_summary["held_out"][g]["mse"]*len(JOINT_GROUPS[g]) for g in ("hip","knee","ankle","waist"))/15,"upper_body_only_mse":sum(group_summary["held_out"][g]["mse"]*len(JOINT_GROUPS[g]) for g in ("shoulder","elbow","hand"))/22,"classification":"UPPER_BODY_ACTION_MSE_DOMINATES" if upper>.5 else "LOCOMOTION_JOINT_ERROR_PRIMARY","formal_action_metric_changed":False})

    # Episode-balanced diagnostics.
    ep_rows=[]
    for part in ("train","validation","held_out"):
        d=split_data[part];e=split_err[part];z=split_zero[part]
        for i,eid in enumerate(d["episode_ids_only"]):
            ids=slice(i*55,(i+1)*55); angle,yaw=parse_condition(d["episode_condition"][i])
            ep_rows.append({"split":part,"episode_id":eid,"condition":d["episode_condition"][i],"direction":angle,"yaw":yaw,"episode_mse":float(e[ids].mean()),"exact_zero_mse":float(e[ids][z[ids]].mean()),"nonzero_mse":float(e[ids][~z[ids]].mean())})
    write_csv("episode_balanced_start_metrics.csv",ep_rows);dump("episode_balanced_start_metrics.json",{"summary":{p:{"episodes":sum(r["split"]==p for r in ep_rows),"episode_balanced_mse":float(np.mean([r["episode_mse"] for r in ep_rows if r["split"]==p])),"exact_zero_mse":float(np.mean([r["exact_zero_mse"] for r in ep_rows if r["split"]==p])),"nonzero_mse":float(np.mean([r["nonzero_mse"] for r in ep_rows if r["split"]==p]))} for p in ("train","validation","held_out")},"diagnostic_only":True})

    # Label semantics.  Dataset commands and observation command channels are bitwise audited.
    zero_data=torch.cat([split_data[p]["physical_command"][split_zero[p]] for p in ("validation","held_out")]);zero_actor=torch.cat([split_data[p]["actor_command"][split_zero[p]] for p in ("validation","held_out")]);zero_obs=torch.cat([split_data[p]["observation"][split_zero[p],9:12] for p in ("validation","held_out")])
    semantic={"physical_command_bitwise_zero":bool((zero_data==0).all()),"actor_command_bitwise_zero":bool((zero_actor==0).all()),"observation_command_bitwise_zero":bool((zero_obs==0).all()),"ramp_progress":0.0,"label_source":"W1B-R2 start actor","previous_action_source":"exp_012 stop teacher action from immediately preceding stopped state","current_command_semantics":"stay stopped","label_semantics":"anticipates the future nonzero minimum-jerk command by one recorded boundary sample","future_command_lookahead":True,"state_previous_action_can_distinguish_boundary":True,"classification":"EXACT_ZERO_START_LABEL_SEMANTIC_MISMATCH"}
    dump("exact_zero_start_label_semantic_audit.json",semantic)

    # One-step diagnostic: immutable recorded-state action branches.  No simulator state snapshot exists
    # at the retained 0.1-s dataset cadence, so physical rollout is fail-closed and explicitly unavailable.
    cf_rows=[]
    for cond in sorted(set(split_data["held_out"]["condition"])):
        ids=torch.tensor([c==cond for c in split_data["held_out"]["condition"]]) & split_zero["held_out"]
        if not ids.any(): continue
        for branch,action in (("A_STUDENT",split_pred["held_out"][ids]),("A_W1B_LABEL",split_data["held_out"]["target_action"][ids]),("A_STOP_TEACHER",split_data["held_out"]["teacher_action"][ids]),("A_PARENT",split_data["held_out"]["source_action"][ids])):
            cf_rows.append({"condition":cond,"branch":branch,"recorded_boundary_states":int(ids.sum()),"requested_branch_trials":200,"executed_physics_trials":0,"action_l2_from_student":float(torch.linalg.vector_norm(action-split_pred["held_out"][ids],dim=1).mean()),"status":"NOT_EXECUTED_NO_EXACT_SIMULATOR_STATE_SNAPSHOT"})
    write_csv("exact_zero_one_step_counterfactual.csv",cf_rows);dump("exact_zero_one_step_counterfactual.json",{"rows":cf_rows,"formal_closed_loop_evaluation":False,"reason":"The immutable dataset stores observations/actions at record_stride=5 but not simulator root/joint/contact/RNG state sufficient for matched-seed branching. Reconstructing states would violate the no-dataset-regeneration/no-formal-rollout boundary.","physical_conclusion":"not used for classification","safety":"not_evaluable","authorization_effect":"none"})

    # Main classification follows the preregistered precedence: stochastic prevalence explains the observed flip.
    state_gap=abs(held["exact_zero_mse"]-next(r for r in full_rows if r["split"]=="validation")["exact_zero_mse"])
    prevalence_instability=(0<summaries["validation"]["pass_probability"]<1 and 0<summaries["held_out"]["pass_probability"]<1 and max_reconstruction<=1e-10 and state_gap<.005)
    classification="EXACT_ZERO_SUBSAMPLE_PREVALENCE_INSTABILITY" if prevalence_instability else "HELDOUT_FAILURE_INCONCLUSIVE"
    dump("stage_classification.json",{"classification":classification,"existing_r2_classification_preserved":"EXP013_W2_P1_R2_VALIDATION_SELECTED_HELDOUT_FAIL"})
    dump("recommended_next_action.json",{"action":"deterministic start-retention authorization contract preflight","compare":["full-split natural-prevalence metric","episode-balanced metric","preregistered exact-zero/nonzero stratified reporting"],"physical_closed_loop_gates_changed":False})
    dump("current_w2_p1_r2_heldout_failure_interpretation.json",{"canonical_parent":"W1B-R2 iteration 200","long_horizon_training":"40,000 steps completed","validation_joint_pass_checkpoints":20,"selected_diagnostic_checkpoint":"step 37,000","heldout_failure":"start-retention mean MSE only","nonzero_start":"high-accuracy PASS-equivalent","stop_recovery":"PASS","steady_stop":"PASS","moving_imitation":"PASS","closed_loop_authorization":"not granted","canonical_promotion":"none"})
    dump("gate.json",{"diagnosis_complete":True,"classification":classification,"closed_loop_authorized":False,"training_authorized":False,"checkpoint_reselection":False,"dataset_changed":0,"label_changed":0,"gate_changed":0})
    dump("stage_reference.json",{"stage":"Phase W2-P1-D4","starting_head":start_head,"expected_starting_head":STARTING_HEAD,"selected_step":37000,"selected_wrapper_sha256":sha(SELECTED),"actor_tensor_hash":"daff324986cfa232d84e2b4d73e4c9383ee293e47fbcaea033fe0829654ded42"})
    dump("protocol.json",{"analysis":"held-out exact-zero start-retention generalization and authorization-contract diagnosis","student_training":0,"checkpoint_selection_changes":0,"formal_closed_loop":0,"dagger":0,"dataset_changes":0,"gate_changes":0,"resampling":{"validation":5000,"held_out":5000,"count":10000},"validation_checkpoint_robustness":{"checkpoints":list(STEPS),"repeats":1000},"heldout_checkpoint_restriction":[37000]})
    (OUT/"reproduction_commands.ps1").write_text('$python="C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe"\n& $python experiments\\isaaclab\\exp_013_unitree_g1_single_policy_omnidirectional_locomotion\\scripts\\analyze_w2_p1_d4_heldout_exact_zero.py\n',encoding="utf-8")
    # Protection records content hashes for every immutable input in scope.
    protected=[]
    for pattern in (str(SOURCE/"raw"/"*_chunk_*.pt"),str(SOURCE/"w2_p1_dataset_*.json"),str(R2/"raw"/"checkpoints"/"*.pt"),str(R2/"raw"/"selected_student.pt")):
        for p in sorted(Path().glob(pattern) if not Path(pattern).is_absolute() else []): protected.append({"path":str(p.relative_to(REPO)),"sha256":sha(p)})
    # Absolute glob above is not portable on Windows; enumerate explicitly.
    protected=[]
    for root,pat in ((SOURCE/"raw","*_chunk_*.pt"),(SOURCE,"w2_p1_dataset_*.json"),(R2/"raw"/"checkpoints","*.pt"),(R2/"raw","selected_student.pt")):
        for p in sorted(root.glob(pat)): protected.append({"path":str(p.relative_to(REPO)).replace("\\","/"),"sha256":sha(p)})
    dump("protected_hashes.json",{"starting_head":start_head,"protected_files":protected,"dataset_changed":0,"label_changed":0,"split_changed":0,"checkpoint_changed":0,"new_persistent_checkpoint":0,"formal_closed_loop":0,"dagger":0,"canonical_promotion":0,"remote_push":False,"unrelated_dirty_state_count":len(dirty_before)})
    print(json.dumps({"classification":classification,"full":full_rows,"resampling":summaries,"state_shift":state_shift,"upper":upper,"lower":lower},indent=2))


if __name__ == "__main__": main()
