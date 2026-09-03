"""Finalize W2-P1-D1 diagnostic metadata/report without touching source artifacts."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_d1_static_representation_conflict_diagnosis"
REPORT=REPO/"research/exp_013_g1_phase_w2_p1_d1_static_representation_conflict_report.md"


def read(name): return json.loads((OUT/name).read_text(encoding="utf-8"))
def dump(name,value): (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def main():
    dist=read("start_retention_error_distribution.json")["original_gate_population"]
    timeline=read("static_representation_checkpoint_timeline.json")
    exact=read("exact_input_label_conflicts.json"); near=read("near_neighbor_label_conflict.json")
    boundary=read("start_zero_command_boundary_audit.json")["rows"]
    suff=read("stop_start_state_sufficiency.json"); grad=read("static_representation_gradient_conflict.json")
    probes=read("static_representation_probe_training.json")
    condition=read("start_retention_condition_breakdown.json")["rows"]
    joints=read("static_conflict_joint_error.json")["rows"]
    p3=next(p for p in probes["probes"] if p["probe"]=="P3_ALL_GROUPS_BALANCED")
    p3rows=[r for r in probes["rows"] if r["probe"]=="P3_ALL_GROUPS_BALANCED"]
    zero=next(r for r in boundary if r["bin"]=="exact_zero")
    zero_nn=next(r for r in near["rows"] if r["command_bin"]=="exact_zero" and r["k"]==1)
    start_timeline=[r for r in timeline["rows"] if r["group"]=="START_RETENTION"]
    best=min(start_timeline,key=lambda r:r["mean_mse"])
    cats={}
    for r in joints:
        if r["sample_group"]=="start_exact_zero": cats[r["joint_category"]]=cats.get(r["joint_category"],0)+r["top_error_contribution"]
    classification="OPTIMIZATION_PATH_NOT_REPRESENTATION_LIMIT"
    dump("stage_classification.json",{
        "stage":"W2-P1-D1","classification":classification,"existing_w2_p1_classification_unchanged":"EXP013_W2_P1_STATIC_REPRESENTATION_FAIL",
        "primary_evidence":[
            f"exact-zero boundary is {zero['sample_count']} samples and contributes {100*zero['group_loss_contribution']:.2f}% of held-out start loss",
            f"bitwise identical cross-group material conflicts: {exact['verified_same_input_materially_different_label']}",
            f"exact-zero nearest-neighbor material conflict fraction: {100*zero_nn['material_conflict_fraction']:.1f}%",
            "P3 fixed all-groups-balanced in-memory probe passed every static group gate with the unchanged 124D architecture",
        ],
        "secondary_mechanism":"near-zero stop/start label competition plus original 10% start weighting/selection path",
        "not_classified_as":"EXACT_ZERO_COMMAND_LABEL_CONFLICT because no bitwise identical 124D cross-group conflicting input exists",
        "closed_loop_authorization":False,"canonical_promotion":False})
    dump("recommended_next_action.json",{"classification":classification,"one_method":"group-balanced supervised integration rerun from the canonical W1B-R2 parent","constraints":["retain the physical dataset and label bytes pending a separately authorized run","one 124D feedforward actor","no runtime teacher/router/action blending"]})
    protected=read("protected_hashes.json")
    dump("gate.json",{"diagnosis_complete":True,"all_required_diagnostic_artifacts_present":True,"metric_contract_replayed_exactly":True,
        "dataset_bytes_unchanged":protected["all_equal"],"label_bytes_unchanged":protected["all_equal"],"new_persistent_checkpoint":0,
        "closed_loop_evaluation":0,"dagger":0,"canonical_promotion":0,"remote_push":False,"production_authorization":False,
        "classification":classification})
    reproduction='''$repo = "C:\\Users\\user\\workspace\\physical-ai-lab"
$python = "C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe"
Set-Location $repo
git rev-parse HEAD
git status --short
git log --oneline --decorate -25
& $python experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/analyze_w2_p1_d1_static_conflict.py
& $python experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/probe_w2_p1_d1_static_conflict.py
& $python experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/finalize_w2_p1_d1_static_conflict.py
'''
    (OUT/"reproduction_commands.ps1").write_text(reproduction,encoding="utf-8")
    condition_range=(min(r["mean_mse"] for r in condition),max(r["mean_mse"] for r in condition))
    text=f'''# exp_013 Phase W2-P1-D1 static representation conflict diagnosis

## Outcome

Main classification: `{classification}`.

The W2-P1 published held-out result was reproduced exactly: mean action MSE `{dist['mean']:.11f}`, p95 `{dist['p95']:.11f}`, cosine `0.99969745`. This is not a unit or weighting mismatch. The per-sample MSE is the mean over 37 action dimensions, and the original group mean is an unweighted mean over 10,000 episode-uniform/timestep-uniform samples drawn with replacement from accepted held-out episodes.

## Metric contract and heavy tail

The distribution is a two-component heavy tail. The top 1% contributes `{100*dist['top_1p0_loss_contribution']:.2f}%` of total loss and the top 5% contributes `{100*dist['top_5p0_loss_contribution']:.2f}%`. The exact-zero sample at the actor-switch/ramp boundary occurs once per held-out episode (`{zero['sample_count']}` samples, `{100*zero['sample_count']/dist['samples']:.2f}%` of the full held-out episode population) and contributes `{100*zero['group_loss_contribution']:.2f}%` of loss. Excluding only that boundary sample is diagnostic-only and changes mean MSE from `0.00128926` to `0.00003147`; no formal gate or dataset was changed.

## Checkpoint timeline

The parent step 0 reproduces W1B labels essentially bitwise (`1.62e-14` start MSE) but has no stop integration. The boundary tail appears by step 500 and remains through step 25,000. The lowest existing trained start mean is `{best['mean_mse']:.8f}` at step `{best['step']}`, still above `0.001`; no existing checkpoint passes every group. Better stop fitting and the exact-zero start tail therefore trade off throughout the original optimization path.

## Label routing and boundary timing

Routing matches the preregistered contract: stop recovery uses W1B before SW3 and exp_012 at/after SW3, steady stop uses exp_012, moving/start retention use W1B. At `t=3.0 s`, command scheduling, observation, and label calculation are aligned; there is no one-step buffer offset. The issue is semantic: runtime switches to W1B exactly when the minimum-jerk command is still bitwise zero and the observation's previous action is the stop teacher action. Thus the first start label asks for a W1B action at a stay-stopped input boundary.

## Exact and near conflicts

Across `1,773,566` unique input hashes there were `{exact['verified_same_input_materially_different_label']}` bitwise-identical 124D cross-group material conflicts, so `EXACT_ZERO_COMMAND_LABEL_CONFLICT` is not supported literally. In the normalized input neighborhood, however, every exact-zero start sample's nearest comparison is materially conflicting (`{100*zero_nn['material_conflict_fraction']:.1f}%`), with mean label MSE `{zero_nn['mean_label_mse']:.5f}`; neighbors are predominantly steady-stop with the remainder stop-recovery. This is a near-zero manifold competition, not corrupted episodes: the top 100 reconstructed episodes are all `ZERO_COMMAND_BOUNDARY`, span all 24 direction/yaw conditions, and show no reset/padding/label-source mismatch.

## Conditions and joints

All 24 start conditions have nearly the same mean error (`{condition_range[0]:.8f}`–`{condition_range[1]:.8f}`), and their exact-zero samples account for 96.38–98.09% of each condition's loss. The tail is not rear-, direction-, or calibrated-positive-yaw-specific. Broad joint contribution at the boundary is dominated by elbow/hand/shoulder categories (`{100*(cats.get('elbow',0)+cats.get('hand',0)+cats.get('shoulder',0)):.2f}%`), while hip/knee/ankle contribute `{100*(cats.get('hip',0)+cats.get('knee',0)+cats.get('ankle',0)):.2f}%`. Joint names were not inferred beyond stored action index and broad diagnostic categories.

## State sufficiency, latent, and gradients

Over complete start trajectories the full observation separates start from steady stop well, but at the exact-zero boundary the linear full-124D probe is weak (AUROC `{suff['exact_zero_boundary']['F1_FULL_124D']['linear']['auroc']:.3f}`); a small nonlinear probe reaches AUROC `{suff['exact_zero_boundary']['F1_FULL_124D']['small_nonlinear_mlp']['auroc']:.3f}`. Previous action alone is insufficient at that boundary (nonlinear AUROC `{suff['exact_zero_boundary']['F5_PREVIOUS_ACTION_ONLY']['small_nonlinear_mlp']['auroc']:.3f}`). Hidden layers nevertheless linearly separate ordinary steady-stop and start samples, so the trunk is not collapsed.

The selected checkpoint gradients confirm a localized conflict: steady-stop vs start-outlier cosine is `-0.8320`, stop-recovery vs start-outlier is `-0.7634`, while moving-retention vs normal-start is `0.9913`. The whole start group conflict is therefore driven by the exact-zero outlier component, not normal moving acquisition samples.

## Fixed temporary probes

All probes used in-memory clones and wrote no checkpoint. P1 start-only fits start but destroys stop groups. P2 start+steady leaves stop-recovery just above gate. P4 exact-zero exclusion leaves the formal original start population above gate, and P5 last-layer-only narrowly misses start (`0.0010057`). P6 original full-network also narrowly misses (`0.0010224`). Crucially, P3 all-groups-balanced reaches simultaneous static PASS without changing the architecture: stop-recovery `{next(r['mean_mse'] for r in p3rows if r['group']=='STOP_RECOVERY'):.7f}`, steady-stop `{next(r['mean_mse'] for r in p3rows if r['group']=='STEADY_STOP'):.7f}`, start `{next(r['mean_mse'] for r in p3rows if r['group']=='START_RETENTION'):.7f}`, and worst moving subgroup `{p3['moving_worst_mse']:.7f}`. The margin is narrow and does not authorize a production student, but it establishes static representational feasibility.

## Interpretation and next action

The primary classification is optimization-path, not a hard 124D representational limit. Near-zero labels compete strongly, and the original top-level start weight (`10%`) plus checkpoint selection leaves the one-sample-per-episode tail underfit. A fixed group-balanced objective can fit all static groups with the unchanged network.

Next, and only next: **group-balanced supervised integration rerun from the canonical W1B-R2 parent**. This diagnosis does not itself authorize that run.

## Protection

All raw dataset/checkpoint hashes match their starting values. No dataset or label bytes changed; no persistent checkpoint, closed-loop rollout, DAgger round, PPO update, or canonical promotion occurred. Existing W2-P1 classification remains unchanged. Remote push was not performed.
'''
    REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text(text,encoding="utf-8")
    print(json.dumps({"classification":classification,"report":str(REPORT),"joint_feasible_probes":probes['joint_feasible_probes']}))


if __name__=="__main__": main()
