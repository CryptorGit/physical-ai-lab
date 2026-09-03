"""Finalize the fail-closed Stage 12 directionality diagnosis and report."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage12_tangential_slip_reward_directionality"
STAGE11 = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage11_tangential_slip_reduction"
REPORT = REPO / "research/exp_011_go2_tangential_slip_reward_directionality_report.md"
RAW_CF = OUT / "raw/counterfactual_0p2_standard.pt"
START = "fed3b08e187b29b5bcbf14e983dd29e60a35b4d4"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


availability = load("low_slip_behavior_availability.json")
temporal = load("slip_reward_temporal_structure.json")
value = load("diagnostic_slip_value_results.json")
advantage = load("slip_advantage_decomposition.json")
gradient = load("actor_gradient_decomposition.json")
minibatch = load("minibatch_gradient_consistency.json")
alignment = load("training_gradient_alignment.json")
ranking = load("within_condition_reward_ranking.json")["results"]
checkpoint_manifest = json.loads((STAGE11 / "checkpoint_manifest.json").read_text(encoding="utf-8"))

# The prohibited-state-write replay could not reproduce a branch in one
# simulator lifecycle. Preserve the failed audit instead of interpreting its
# perturbation outcomes.
counterfactual = torch.load(RAW_CF, map_location="cpu", weights_only=False)
matching = counterfactual["matching"]
error_fields = ("root", "joint", "previous_action", "contact_age", "heading_state")
max_errors = {field: max(float(row[field]) for row in matching) for field in error_fields}
valid_variants = sum(bool(row["matched"]) for row in matching)
attempted_states = int(counterfactual["episodes"])
validity = "COUNTERFACTUAL_PREBRANCH_MATCH_FAIL"

dump("counterfactual_branch_manifest.json", {
    "requested": {
        "speeds_m_s": [0.2, 0.4, 0.6, 1.2, 2.0],
        "states_per_speed": 100,
        "total_states": 500,
        "action_dimensions": 12,
        "primary_delta": 0.02,
    },
    "executed_before_fail_closed": {
        "speed_m_s": 0.2,
        "attempted_states": attempted_states,
        "attempted_variants": len(matching),
        "valid_variants": valid_variants,
    },
    "selection": "first stable-contact state in fixed 2.0-2.5 s window",
    "replay": "same reset seed, command, deterministic policy prefix; ordinary reset only",
    "state_setter": False,
    "teleport": False,
    "state_injection": False,
    "status": validity,
    "remaining_conditions": "NOT_EXECUTED_AFTER_VALIDITY_FAILURE",
})
dump("prebranch_matching_audit.json", {
    "gate_tolerance": 1.0e-5,
    "all_variants_matched": False,
    "valid_variants": valid_variants,
    "attempted_variants": len(matching),
    "maximum_absolute_error": max_errors,
    "fresh_process_baseline_repeat": {
        "matched": True,
        "maximum_absolute_error": 0.0,
        "interpretation": (
            "A fresh Isaac process reproduced the baseline prefix bitwise, "
            "but ordinary repeated reset inside one PhysX lifecycle retained "
            "uncontrolled contact/runtime history."
        ),
    },
    "fail_closed_reason": (
        "The permitted workflow forbids state injection; therefore no "
        "perturbation outcome was accepted when its pre-branch state differed."
    ),
})
dump("counterfactual_action_results.json", {
    "status": "NOT_EVALUABLE",
    "reason": validity,
    "accepted_rows": 0,
    "invalid_rows_excluded": len(counterfactual["rows"]),
    "horizons_steps": [1, 2, 4, 8],
})
not_evaluable = {
    "status": "NOT_EVALUABLE",
    "reason": validity,
    "classification": "COUNTERFACTUAL_VALIDITY_FAILURE",
}
dump("local_slip_controllability.json", not_evaluable | {
    "requested_gate": {
        "SLIP_LOCALLY_CONTROLLABLE": ">=30% improving states at every major speed",
        "SLIP_NOT_LOCALLY_CONTROLLABLE": "<10% overall",
    },
})
dump("local_tradeoff_analysis.json", not_evaluable | {
    "speed": "NOT_EVALUABLE", "heading": "NOT_EVALUABLE",
    "contact": "NOT_EVALUABLE", "flight": "NOT_EVALUABLE",
})
dump("reward_controllability_agreement.json", not_evaluable | {
    "reward_problem_state_identification": "SUPPORTED_BY_WITHIN_CONDITION_RANKING",
    "improving_action_agreement": "NOT_EVALUABLE",
})

classification = "SLIP_REWARD_DIRECTIONALITY_INCONCLUSIVE"
dump("stage12_classification.json", {
    "classification": classification,
    "precedence_trigger": "counterfactual / telemetry validity failure",
    "primary_evidence": validity,
    "secondary_findings_not_promoted_to_causal_classification": [
        availability["classification"],
        temporal["classification"],
        gradient["strength_classification"],
        gradient["conflict_classification"],
        minibatch["classification"],
    ],
    "protected_stage11_result": "GO2_TANGENTIAL_SLIP_NO_EFFECT",
})
dump("pilot_readiness.json", {
    "classification": "PILOT2_NOT_READY",
    "gradient_calibrated_weight_allowed": False,
    "failed_requirements": [
        "counterfactual pre-branch identity",
        "local controllability PASS",
        "capability-conflict exclusion",
    ],
})
dump("gradient_calibrated_weight_proposal.json", {
    "status": "NOT_APPLICABLE",
    "current_lambda": 0.00559195994498,
    "reason": (
        "Weight proposal is only permitted for "
        "SLIP_REWARD_DIRECTIONAL_BUT_UNDERWEIGHTED; local controllability "
        "was not validly established and base/slip gradients conflict."
    ),
})
dump("recommended_next_action.json", {
    "action": "establish a reproducible no-state-injection counterfactual replay contract",
    "pilot2": False,
    "single_change": True,
    "reason": validity,
})

previous_protection = json.loads((STAGE11 / "protected_hashes.json").read_text(encoding="utf-8"))
official = REPO / (
    ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/"
    "Assets/Isaac/6.0/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/"
    "Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt"
)
stage4 = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage4_resumed_optimizer_training/checkpoints/model_50.pt"
stage7 = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage7_low_speed_gait_stabilization/checkpoints/model_50.pt"
stage10_controller = EXP / "src/go2_bidirectional/phase_gated_heading.py"
protected = {
    "starting_head": START,
    "exp_005_to_exp_010": "UNCHANGED",
    "exp_011_stage1_to_stage11": "UNCHANGED",
    "capability_manifest": "UNCHANGED",
    "production_artifact": "UNCHANGED",
    "isaac_lab_core": "UNCHANGED",
    "official_parent": {
        "expected": "32039715f0892650691aa8d5c50233e7c4b858469d87114ef92794e2c65b59c0",
        "actual": sha(official),
    },
    "stage4_selected": {
        "expected": "e2a3de144984683efcc7b4fe451898c3d2b450a7ae3696ad6784a027a9756bea",
        "actual": sha(stage4),
    },
    "stage7_selected": {
        "expected": "d6bb5b7be94f0a827576256b6ae420cdc5b2267c389c7ba92951801f9e2899bd",
        "actual": sha(stage7),
    },
    "stage11_checkpoints": [
        {
            "iteration": item["iteration"],
            "expected": item["sha256"],
            "actual": sha(item["path"]),
        }
        for item in checkpoint_manifest["checkpoints"]
    ],
    "stage10_heading_controller": {
        "expected": previous_protection["stage10_heading_controller"]["expected"],
        "actual": sha(stage10_controller),
    },
    "go2_endpoint_evaluation_v1": previous_protection["go2_endpoint_evaluation_v1"],
    "go2_tangential_slip_evaluation_v1": {
        "expected": "74b46a8ed230d4531259ff1ec52ef9937d308ec3a1334b9feeaa5a10707d0f83",
        "actual": json.loads((STAGE11 / "slip_evaluation_protocol_hash.json").read_text(encoding="utf-8"))["sha256"],
    },
    "production_ppo_update": 0,
    "reward_optimization": 0,
    "remote_push": False,
}
for key in ("official_parent", "stage4_selected", "stage7_selected", "stage10_heading_controller",
            "go2_tangential_slip_evaluation_v1"):
    protected[key]["unchanged"] = protected[key]["expected"] == protected[key]["actual"]
for item in protected["stage11_checkpoints"]:
    item["unchanged"] = item["expected"] == item["actual"]
dump("protected_hashes.json", protected)
dump("gate.json", {
    "stage12_complete": True,
    "fixed_rollout_complete": True,
    "reward_ranking_complete": True,
    "diagnostic_value_reliable": value["reliability"] == "RELIABLE",
    "gradient_decomposition_complete": True,
    "counterfactual_valid": False,
    "fail_closed": True,
    "classification": classification,
    "pilot_readiness": "PILOT2_NOT_READY",
    "production_ppo_update": 0,
    "reward_optimization": 0,
})

reproduction = r'''$ErrorActionPreference = "Stop"
Set-Location "$HOME\workspace\physical-ai-lab"
$exp = ".\experiments\isaaclab\exp_011_unitree_go2_bidirectional_speed_transitions"
$isaac = "$HOME\workspace\IsaacLab\isaaclab.bat"

python "$exp\scripts\prepare_stage12.py"
& $isaac -p "$exp\scripts\collect_stage12_rollout.py" --family steady --condition-index 0 --device cuda:0 --headless
& $isaac -p "$exp\scripts\analyze_stage12_offline.py"
& $isaac -p "$exp\scripts\gradient_stage12.py" --device cuda:0 --headless
& $isaac -p "$exp\scripts\counterfactual_stage12.py" --speed-index 0 --mode standard --device cuda:0 --headless
& $isaac -p "$exp\scripts\finalize_stage12.py"

# Counterfactual replay intentionally stops fail-closed when pre-branch
# identity does not satisfy 1e-5; no production PPO update is performed.
'''
(OUT / "reproduction_commands.ps1").write_text(reproduction, encoding="utf-8")

low = ranking["LOWEST_10"]
high = ranking["HIGHEST_10"]
trajectory = alignment["trajectory"]
report = f"""# exp_011 Go2 tangential-slip reward directionality diagnosis

## 結論

Stage 12の主要分類は **{classification}**、Pilot readinessは
**PILOT2_NOT_READY** である。固定rollout上ではrewardの順位付け、短い時間
lag、学習可能なslip value、弱いが一貫したactor gradientを確認した。しかし
禁止されたstate injectionを使わない同一process replayではpre-branch状態が
一致せず、局所action方向と能力trade-offを物理的に検証できなかった。この
validity failureをprecedence 1としてfail-closedした。

## Reward ranking

同じspeed・schedule phase・4足contact pattern内で0.2秒segmentを順位付けした。
lowest-slip decileのraw score平均は {low['raw_slip_score']['mean']:.4f}、
highest-slip decileは {high['raw_slip_score']['mean']:.4f} だった。対応する
speed MAE平均は {low['speed_mae']['mean']:.4f} / {high['speed_mae']['mean']:.4f}
m/s、base reward平均は {low['base_reward']['mean']:.5f} /
{high['base_reward']['mean']:.5f}。主要5速度のsafe low-slip segment比率は
{', '.join(f"{speed}: {item['safe_rate_all_segments']*100:.1f}%"
            for speed, item in availability['by_speed'].items())}で、
**{availability['classification']}** と判定した。

## Temporal structure

action-rate変化との最大相関はlag {temporal['best_lag']['lag_steps']} step
（{temporal['best_lag']['lag_s']:.2f}s）、相関
{temporal['best_lag']['action_change_to_slip_correlation']:.3f}で、
**{temporal['classification']}**。固定trajectoryからStage 11と同じ
discountでslip-only returnを計算した。

## Advantage and gradient

48→128→128→1 ELUの診断V_slipはepisode/seed 70/15/15 splitでtest R²
{value['test']['r2']:.3f}、MAE {value['test']['mae']:.6f}。診断GAEの
A_slip stdは {advantage['A_slip']['std']:.6f}、A_base stdは
{advantage['A_base']['std']:.3f}。初期actor固定batch上で
|g_slip|={gradient['norms']['g_slip']:.6f}、
|g_base|={gradient['norms']['g_base']:.6f}、
q_g={gradient['norms']['q_g']:.6f}（{gradient['strength_classification']}）。
base/slip cosineは {gradient['cosines']['base_slip']:.3f}
（{gradient['conflict_classification']}）。100 fixed permutationsの
pairwise slip-gradient cosine中央値は
{minibatch['pairwise_slip_gradient_cosine_median']:.3f}
（{minibatch['classification']}）。

Stage 11 checkpoint軌跡ではiteration 1/10/25の初期方策差がslip方向と負、
後半の一部が正だったが、validation precedenceを改善せずiteration 0が選択
された。これは「gradientが弱くbase側に支配された」ことと整合するが、
counterfactual failureのためunderweightingを因果分類には昇格しない。

## Local controllability

0.2m/sの100 branch stateでordinary resetによるsame-seed replayを監査した。
fresh process間のbaseline prefixはbitwise一致した一方、同一PhysX lifecycle
の再resetではroot最大差 {max_errors['root']:.3f}、joint
{max_errors['joint']:.3f}、previous action {max_errors['previous_action']:.3f}、
contact age {max_errors['contact_age']:.1f}となり、許容1e-5を満たすvariantは
{valid_variants}/{len(matching)}だった。残り400 stateとlinearity replayは
結果を作るために続行せず停止した。state setter、teleport、別env state copy
は0である。

したがってlocally improving state rate、joint/leg/phase別controllability、
speed/heading/contact Pareto trade-offは **NOT_EVALUABLE**。invalid branchの
perturbation結果は解析から除外した。

## Classification and next action

- Classification: **{classification}**
- Pilot readiness: **PILOT2_NOT_READY**
- Next: **establish a reproducible no-state-injection counterfactual replay contract**

gradient-calibrated weightは提案しない。local controllability PASSとcapability
conflict不在を確認できていないため、weight増加は許可されない。

## Protection

Stage 1〜11、公式/Stage 4/Stage 7/Stage 11 checkpoint、Stage 10 controller、
両評価protocol、capability manifest、production artifact、Isaac Lab coreは
変更していない。production PPO update=0、reward optimization=0、remote
push=false。
"""
REPORT.write_text(report, encoding="utf-8")

head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
print(json.dumps({
    "classification": classification,
    "pilot_readiness": "PILOT2_NOT_READY",
    "head": head,
    "q_g": gradient["norms"]["q_g"],
    "base_slip_cosine": gradient["cosines"]["base_slip"],
    "counterfactual_valid": False,
}, indent=2))
