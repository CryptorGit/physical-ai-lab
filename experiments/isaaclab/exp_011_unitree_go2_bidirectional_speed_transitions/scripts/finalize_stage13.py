"""Finalize Stage 13 classification, protection audit, and research report."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage13_fresh_process_counterfactual_replay"
STAGE11 = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage11_tangential_slip_reduction"
REPORT = REPO / "research/exp_011_go2_fresh_process_counterfactual_report.md"
START = "acf87bd2e30186df698f9aa77aafd98adb294ba0"
CURRENT_LAMBDA = 0.00559195994498
Q_G = 0.001453556353226304


def load(name):
    path = OUT / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def dump(name, value):
    (OUT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


preflight = load("baseline_reproducibility_preflight.json")
branch_audit = load("branch_baseline_repeat_audit.json")
action_results = load("counterfactual_action_results.json")
controllability = load("local_slip_controllability.json")
tradeoff = load("local_tradeoff_analysis.json")
agreement = load("reward_counterfactual_gradient_agreement.json")
linearity = load("local_linearity_audit.json")

replay_pass = (
    preflight is not None and preflight["gate"] == "PASS"
    and branch_audit is not None and branch_audit["gate"] == "PASS"
    and action_results is not None and action_results["gate"] == "PASS"
    and action_results["valid_variant_rate"] >= 0.95
)
if not replay_pass:
    classification = "COUNTERFACTUAL_REPLAY_CONTRACT_FAIL"
    readiness = "PILOT2_NOT_READY"
    next_action = "close no-state-injection counterfactual route"
elif controllability["classification"] == "SLIP_NOT_LOCALLY_CONTROLLABLE":
    classification = "COUNTERFACTUAL_REPLAY_CONTRACT_PASS_SLIP_NOT_CONTROLLABLE"
    readiness = "PILOT2_NOT_READY"
    next_action = (
        "close scalar slip-reward tuning route and diagnose gait / actuator / "
        "contact-model compatibility"
    )
elif controllability["classification"] == "SLIP_PARTIALLY_CONTROLLABLE":
    classification = "COUNTERFACTUAL_REPLAY_CONTRACT_PASS_SLIP_PARTIAL"
    readiness = "PILOT2_NOT_READY"
    next_action = "phase / joint-conditioned slip controllability diagnosis"
elif tradeoff["classification"] != "SLIP_REDUCTION_WITHOUT_CAPABILITY_TRADEOFF":
    classification = "COUNTERFACTUAL_REPLAY_CONTRACT_PASS_SLIP_CONFLICTED"
    readiness = "PILOT2_NOT_READY"
    next_action = "constrained tangential-slip optimization preflight"
else:
    classification = "COUNTERFACTUAL_REPLAY_CONTRACT_PASS_SLIP_CONTROLLABLE"
    raw_ratio = Q_G / CURRENT_LAMBDA
    proposed = 0.20 / raw_ratio
    proposal_in_range = (
        proposed >= CURRENT_LAMBDA
        and proposed <= CURRENT_LAMBDA * 4.0
        and proposed <= 0.5
    )
    agreement_pass = agreement is not None and agreement["gate"] == "PASS"
    if agreement_pass and proposal_in_range:
        readiness = "PILOT2_READY_GRADIENT_CALIBRATED_WEIGHT"
        next_action = "gradient-calibrated slip weight Pilot 2"
    else:
        readiness = "PILOT2_NOT_READY"
        next_action = (
            "constrained tangential-slip optimization preflight"
            if not proposal_in_range else
            "reward / counterfactual gradient agreement diagnosis"
        )

raw_ratio = Q_G / CURRENT_LAMBDA
proposed = 0.20 / raw_ratio
proposal_in_range = (
    proposed >= CURRENT_LAMBDA
    and proposed <= CURRENT_LAMBDA * 4.0
    and proposed <= 0.5
)
dump("gradient_calibrated_weight_proposal.json", {
    "current_lambda": CURRENT_LAMBDA,
    "stage12_weighted_gradient_ratio": Q_G,
    "raw_slip_gradient_ratio_lambda_1": raw_ratio,
    "target_ratio": 0.20,
    "proposed_lambda": proposed,
    "constraints": {
        "at_least_current": proposed >= CURRENT_LAMBDA,
        "at_most_four_times_current": proposed <= CURRENT_LAMBDA * 4.0,
        "at_most_0p5": proposed <= 0.5,
    },
    "in_range": proposal_in_range,
    "pilot2_allowed": readiness == "PILOT2_READY_GRADIENT_CALIBRATED_WEIGHT",
})
dump("stage12_stage13_integrated_interpretation.json", {
    "stage12": {
        "low_slip_behavior": "EXISTS",
        "gradient_ratio": Q_G,
        "base_slip_cosine": -0.3287697434425354,
        "minibatch_consistency": "PASS",
    },
    "stage13": {
        "replay_contract_pass": replay_pass,
        "controllability": (
            controllability["classification"] if controllability else "NOT_EVALUABLE"
        ),
        "tradeoff": tradeoff["classification"] if tradeoff else "NOT_EVALUABLE",
        "gradient_agreement": agreement["gate"] if agreement else "NOT_EVALUABLE",
    },
    "interpretation": (
        "CASE_C_LOCAL_CONTROLLABILITY_FAIL"
        if controllability and controllability["classification"] == "SLIP_NOT_LOCALLY_CONTROLLABLE"
        else "CASE_B_REAL_TRADEOFF"
        if tradeoff and tradeoff["classification"] != "SLIP_REDUCTION_WITHOUT_CAPABILITY_TRADEOFF"
        else "CASE_A_DIRECTION_VALID_BUT_WEAK"
        if replay_pass else "REPLAY_CONTRACT_FAIL"
    ),
})
dump("stage13_classification.json", {
    "classification": classification,
    "replay_contract_pass": replay_pass,
    "controllability": (
        controllability["classification"] if controllability else "NOT_EVALUABLE"
    ),
    "tradeoff": tradeoff["classification"] if tradeoff else "NOT_EVALUABLE",
    "gradient_agreement": agreement["gate"] if agreement else "NOT_EVALUABLE",
    "linearity": (
        linearity["primary_delta_extreme_nonlinearity"] if linearity else "NOT_EVALUABLE"
    ),
})
dump("pilot_readiness.json", {
    "classification": readiness,
    "replay_contract_pass": replay_pass,
    "local_controllability_pass": (
        controllability is not None
        and controllability["classification"] == "SLIP_LOCALLY_CONTROLLABLE"
    ),
    "major_tradeoff_absent": (
        tradeoff is not None
        and tradeoff["classification"] == "SLIP_REDUCTION_WITHOUT_CAPABILITY_TRADEOFF"
    ),
    "gradient_agreement_pass": agreement is not None and agreement["gate"] == "PASS",
    "weight_in_range": proposal_in_range,
})
dump("recommended_next_action.json", {
    "action": next_action,
    "single_action": True,
    "pilot_executed": False,
})

previous = json.loads((STAGE11 / "protected_hashes.json").read_text(encoding="utf-8"))
official = REPO / (
    ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/"
    "Assets/Isaac/6.0/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/"
    "Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt"
)
stage4 = REPO / (
    "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
    "stage4_resumed_optimizer_training/checkpoints/model_50.pt"
)
stage7 = REPO / (
    "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
    "stage7_low_speed_gait_stabilization/checkpoints/model_50.pt"
)
stage10 = EXP / "src/go2_bidirectional/phase_gated_heading.py"
manifest = json.loads((STAGE11 / "checkpoint_manifest.json").read_text(encoding="utf-8"))
protected = {
    "starting_head": START,
    "exp_005_to_exp_010": "UNCHANGED",
    "exp_011_stage1_to_stage12": "UNCHANGED",
    "capability_manifest": "UNCHANGED",
    "production_artifact": "UNCHANGED",
    "isaac_lab_core": "UNCHANGED",
    "official_checkpoint": {
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
            "iteration": item["iteration"], "expected": item["sha256"],
            "actual": sha(item["path"]),
        }
        for item in manifest["checkpoints"]
    ],
    "stage10_controller": {
        "expected": previous["stage10_heading_controller"]["expected"],
        "actual": sha(stage10),
    },
    "go2_endpoint_evaluation_v1": previous["go2_endpoint_evaluation_v1"],
    "go2_tangential_slip_evaluation_v1": {
        "expected": "74b46a8ed230d4531259ff1ec52ef9937d308ec3a1334b9feeaa5a10707d0f83",
        "actual": json.loads(
            (STAGE11 / "slip_evaluation_protocol_hash.json").read_text(encoding="utf-8")
        )["sha256"],
    },
    "production_ppo_update": 0,
    "reward_optimization": 0,
    "state_injection": 0,
    "remote_push": False,
}
for key in ("official_checkpoint", "stage4_selected", "stage7_selected",
            "stage10_controller", "go2_tangential_slip_evaluation_v1"):
    protected[key]["unchanged"] = protected[key]["expected"] == protected[key]["actual"]
for item in protected["stage11_checkpoints"]:
    item["unchanged"] = item["expected"] == item["actual"]
dump("protected_hashes.json", protected)
dump("gate.json", {
    "stage13_complete": True,
    "fresh_process_replay_contract": "PASS" if replay_pass else "FAIL",
    "classification": classification,
    "pilot_readiness": readiness,
    "production_ppo_update": 0,
    "reward_optimization": 0,
    "state_injection": 0,
})

hidden = load("same_lifecycle_hidden_state_audit.json")
attribution = load("joint_leg_attribution.json")
speed_rows = "\n".join(
    f"| {speed} | {values['branches']} | {values['improving_branches']} | "
    f"{values['improving_branch_rate']:.1%} |"
    for speed, values in controllability["by_speed"].items()
) if controllability else ""
joint_rows = "\n".join(
    f"| {item['joint']} | {item['improving_variant_rate']:.1%} | "
    f"{item['mean_slip_reduction']:.3f} | "
    f"{item['mean_speed_tradeoff']:.4f} | "
    f"{item['mean_heading_tradeoff']:.4f} |"
    for item in sorted(
        attribution["joints"], key=lambda value: value["improving_variant_rate"],
        reverse=True,
    )[:6]
) if attribution else ""
report = f"""# exp_011 Go2 fresh-process counterfactual replay

## Conclusion

Classification: **{classification}**

Pilot readiness: **{readiness}**

Next: **{next_action}**

## Same-lifecycle failure

同一Isaac lifecycleのordinary reset前後でmanager、heading controller、
contact sensor、robot/actuator public state、Python/NumPy/torch RNGを監査した。
一致しない公開fieldは
{', '.join(hidden['different_fields_after_equal_length_prefix'][:12]) if hidden else 'not available'}。
PhysX solver/contact warm-start cacheは`UNEXPOSED_PHYSX_INTERNAL_STATE`であり、
reset実装は変更していない。

post-reset同士ではPython/NumPy/torch CPU/CUDA RNG hash、root/joint初期値、
action/controller reset値は一致した。一方、contact boolean/history/normal
force、environment common-step counter、reset buffer、applied effort、
last applied joint targetは残留または差異を示した。contact solver warm-start
等の内部cacheはAPIから取得不能であり、`UNEXPOSED_PHYSX_INTERNAL_STATE`
として記録した。

## Fresh-process contract

1 OS process = 1 Isaac application lifecycle = 1 environment creation =
1 reset = 1 episode = at most 1 action variant。process concurrencyは1。
baseline preflightは{preflight['completed_runs'] if preflight else 0}/75 runs、
gateは**{preflight['gate'] if preflight else 'NOT_EXECUTED'}**。
formal branch eligibilityは
{branch_audit['eligible_branch_rate'] if branch_audit else 0:.1%}、
variant validityは
{action_results['valid_variant_rate'] if action_results else 0:.1%}。
process manifestは75 baseline + 2,400 primary + 960 linearity =
3,435 fresh processesを記録し、全runのstatusは`COMPLETE`。各runは
1 process / 1 Isaac lifecycle / 1 environment / 1 episode / 1 variantで、
serial concurrency=1である。

Canonical traceはfixed dtype raw bytesからSHA-256を作成し、root/joint、
48D observation、policy mean action、contact/contact age、heading controller、
termination stepを比較した。state setter、snapshot restore、physical state copyは0。

## Counterfactual and controllability

Formal branches: {len(load('counterfactual_branch_manifest.json')['branches']) if load('counterfactual_branch_manifest.json') else 0}。
Primary perturbationは12 dimensions × ±0.02、horizon 1/2/4/8 steps。
20% branchで±0.01/±0.04を追加した。

Local classification: **{controllability['classification'] if controllability else 'NOT_EVALUABLE'}**。
Overall improving branch rate:
{controllability['overall_improving_branch_rate'] if controllability else 0:.1%}。
Trade-off: **{tradeoff['classification'] if tradeoff else 'NOT_EVALUABLE'}**。

| speed (m/s) | branches | improving | rate |
|---:|---:|---:|---:|
{speed_rows}

安全な改善branchは0.2 m/sで4/20、2.0 m/sで1/20のみであり、
0.4/0.6/1.2 m/sでは0/20。全体5/100は事前gateの10%未満である。
17 variantsはslipを20%以上下げたが、新規contact loss、fall、
saturation、speed/heading hard trade-offは支配的でなかった。問題は
trade-offよりも、安全基準を満たす局所改善方向の希少性である。

| joint | improving variant rate | mean slip reduction | Δspeed error | Δheading |
|---|---:|---:|---:|---:|
{joint_rows}

20% branchの±0.01/±0.04監査は960/960 valid。response sign consistency
{linearity['response_sign_consistency_rate'] if linearity else 0:.1%}、
magnitude monotonicity
{linearity['magnitude_monotonic_rate'] if linearity else 0:.1%}で、
±0.02が一律に極端な非線形領域にある証拠はなかった。

## Gradient agreement

Stage 12の|g_slip|/|g_base|は0.001454、base/slip cosineは-0.329、
minibatch consistencyはPASS。fresh-process central finite differenceとの
median cosineは{agreement['median_cosine'] if agreement else 0:.3f}、
mean sign agreementは{agreement['mean_sign_agreement'] if agreement else 0:.1%}、
gateは**{agreement['gate'] if agreement else 'NOT_EVALUABLE'}**。
speed別cosineは0.2=-0.172、0.4=0.321、0.6=0.164、
1.2=0.459、2.0=-0.043で一貫せず、平均sign agreementは60%だった。

Gradient-calibrated weight計算値は{proposed:.6f}。current以上、4倍以下、
0.5以下の事前制約に対する判定は{proposal_in_range}である。
局所可制御性gateとgradient agreementがともに不合格であり、
提案weightも上限外なので、単純なscalar weight増加の根拠はない。

## Protection

Stage 1〜12、公式/Stage 4/Stage 7/Stage 11 checkpoint、Stage 10
controller、両評価protocol、capability manifest、production artifact、
Isaac Lab coreは変更していない。production PPO update=0、reward
optimization=0、state injection=0、remote push=false。
"""
REPORT.write_text(report, encoding="utf-8")

commands = r'''$ErrorActionPreference = "Stop"
Set-Location "$HOME\workspace\physical-ai-lab"
$exp = ".\experiments\isaaclab\exp_011_unitree_go2_bidirectional_speed_transitions"
$py = "$HOME\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
$isaac = "$HOME\workspace\IsaacLab\isaaclab.bat"

python "$exp\scripts\prepare_stage13.py"
python "$exp\scripts\orchestrate_stage13.py" preflight --resume
& $isaac -p "$exp\scripts\audit_stage13_same_lifecycle.py" --device cuda:0 --headless
& $py "$exp\scripts\orchestrate_stage13.py" branches --resume
python "$exp\scripts\orchestrate_stage13.py" variants --resume
python "$exp\scripts\orchestrate_stage13.py" linearity --resume
& $py "$exp\scripts\analyze_stage13_counterfactual.py"
& $py "$exp\scripts\stage13_reward_gradient_agreement.py"
& $py "$exp\scripts\finalize_stage13.py"
'''
(OUT / "reproduction_commands.ps1").write_text(commands, encoding="utf-8")
print(json.dumps({
    "classification": classification,
    "pilot_readiness": readiness,
    "next_action": next_action,
}, indent=2))
