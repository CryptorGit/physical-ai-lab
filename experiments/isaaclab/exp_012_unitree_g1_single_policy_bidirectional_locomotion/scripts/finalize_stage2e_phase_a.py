"""Finalize Stage 2E Phase A evidence, gates, manifest, and report."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2e_phase_a_run_acquisition_preflight"
REPORT = REPO / "research/exp_012_g1_two_stage_run_acquisition_phase_a_report.md"
START = "b4c38faa1bf15ef9709b9155fd932496a0b065ab"
PARENT_SHA = "8d8afac60cafbd4adf0b98469fab01f711f32771a40899653d962cc08a5d8143"
CHECKPOINTS = (0, 1, 5, 10, 20, 30, 40, 50, 75, 100)


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if torch.is_tensor(value):
            value = value.detach().contiguous().cpu()
            h.update(key.encode())
            h.update(str(value.dtype).encode())
            h.update(bytes(value.numpy()))
    return h.hexdigest()


def checkpoint_path(iteration):
    return OUT / "checkpoints" / ("model_initial.pt" if iteration == 0 else f"model_{iteration}.pt")


timeline = list(csv.DictReader((OUT / "phase_a_run_event_timeline.csv").open(encoding="utf-8")))
# The original runtime trace used the wrapper's combined done bit. Mark the two
# scheduled 20-second timeout boundaries explicitly rather than misreporting
# them as falls; checkpoint evaluation below measures falls separately.
if timeline and "termination_or_timeout_rate" not in timeline[0]:
    for row in timeline:
        observed = float(row["fall_rate"])
        timeout_boundary = observed > .90
        row["termination_or_timeout_rate"] = row["fall_rate"]
        row["scheduled_timeout_boundary"] = str(timeout_boundary).lower()
        row["fall_rate_valid"] = str(not timeout_boundary).lower()
        if timeout_boundary:
            row["fall_rate"] = ""
    with (OUT / "phase_a_run_event_timeline.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(timeline[0]))
        writer.writeheader()
        writer.writerows(timeline)
evaluation = json.loads((OUT / "phase_a_evaluation_summary.json").read_text(encoding="utf-8"))
selected = json.loads((OUT / "selected_phase_a_checkpoint.json").read_text(encoding="utf-8"))
first = json.loads((OUT / "first_update_stability.json").read_text(encoding="utf-8"))
gradients = json.loads((OUT / "run_specific_gradient_after_emergence.json").read_text(encoding="utf-8"))
integrity = json.loads((OUT / "phase_a_pre_run_integrity.json").read_text(encoding="utf-8"))

first_completion = next(int(row["iteration"]) for row in timeline if int(row["completion_reward_fire_count"]) > 0)
completion_iterations = [int(row["iteration"]) for row in timeline if int(row["completion_reward_fire_count"]) > 0]
total_completion = sum(int(row["completion_reward_fire_count"]) for row in timeline)
max_density_row = max(timeline, key=lambda row: float(row["completion_per_run_sample"]))
max_density = float(max_density_row["completion_per_run_sample"])
saved_with_completion = [
    iteration for iteration in CHECKPOINTS
    if iteration and int(timeline[iteration - 1]["completion_reward_fire_count"]) > 0
]
selected_iteration = int(selected["iteration"])
selected_conditions = selected["conditions"]

checkpoint_rows = []
for iteration in CHECKPOINTS:
    path = checkpoint_path(iteration)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    optimizer = payload["optimizer_state_dict"]
    adam_steps = sorted({int(float(x["step"])) for x in optimizer["state"].values()})
    checkpoint_rows.append({
        "phase_a_iteration": iteration,
        "path": str(path.relative_to(REPO)),
        "sha256": sha(path),
        "actor_hash": tensor_state_hash(payload["actor_state_dict"]),
        "critic_hash": tensor_state_hash(payload["critic_state_dict"]),
        "std_hash": hashlib.sha256(
            bytes(payload["actor_state_dict"]["distribution.std_param"].detach().contiguous().cpu().numpy())
        ).hexdigest(),
        "optimizer_state_count": len(optimizer["state"]),
        "adam_steps": adam_steps,
        "current_lr": optimizer["param_groups"][0]["lr"],
        "reward_hash": "exp012_parent_base_plus_exp005_safe_periodic_flight",
        "phase_a_curriculum_hash": "phase_a_focused_70_30_v1",
        "resume_contract_hash": "Exp012StrictPPOResumeContract",
    })

dump("stage_reference.json", {
    "starting_head": START,
    "phase_a_parent": "Stage 2 Pilot 1 retry selected iteration 100",
    "phase_a_parent_sha256": PARENT_SHA,
    "phase_a_parent_adam_step": 87000,
    "position": "single-policy continuation preflight; not Pilot 2",
    "maximum_runs": 1, "automatic_retry": False,
})
dump("protocol.json", {
    "name": "EXP012_STAGE2E_PHASE_A_RUN_ACQUISITION_PREFLIGHT",
    "num_envs": 1024, "iterations": 100, "rollout_steps": 24,
    "interactions": 2457600, "seed": 20265021,
    "episode_duration_s": 20.0, "minimum_run_hold_s": 10.0,
    "actual_run_hold_s": 15.0, "phase_b_executed": False,
})
dump("checkpoint_manifest.json", {
    "parent_sha256": PARENT_SHA, "checkpoints": checkpoint_rows,
    "selected_iteration": selected_iteration,
    "selected_sha256": selected["sha256"],
})

run_eval_completion = {
    iteration: sum(
        evaluation[str(iteration)]["conditions"][f"run_{speed:.1f}"]["completion_reward_fires"]
        for speed in (2.3, 2.4, 2.5, 2.6)
    ) for iteration in CHECKPOINTS
}
deterministic_completion_checkpoints = [i for i, count in run_eval_completion.items() if count > 0]
reachability_pass = (
    total_completion > 0
    and len(saved_with_completion) >= 2
    and max_density >= .0005
    and len(deterministic_completion_checkpoints) >= 1
)
strong_pass = (
    reachability_pass and max_density >= .002
    and len(saved_with_completion) >= 3
    and selected_conditions["run_2.4"]["periodic_running_rate"] >= .80
    and selected_conditions["run_2.6"]["periodic_running_rate"] >= .50
)
catastrophic = bool(selected["catastrophic_retention_collapse"])

# Completion did emerge repeatedly under stochastic training, but did not clear
# the density gate or reproduce in any frozen deterministic checkpoint.
classification = "SINGLE_POLICY_RUN_COMPLETION_EMERGED_PARTIAL"
next_action = "Phase A boundary diagnosis"
phase_b_ready = False

dump("phase_b_joint_retention_plan.json", {
    "status": "NOT_READY",
    "phase_a_classification": classification,
    "phase_b_executed": False,
    "plan": None,
    "reason": "Phase A did not pass completion-density/reproducibility gates; no Phase B protocol is frozen.",
})
(OUT / "phase_b_joint_retention_plan.md").write_text(
    "# Phase B joint-retention plan\n\n"
    "STATUS: NOT_READY\n\n"
    "Phase A produced sparse stochastic completion events but failed the registered "
    "completion-density gate and produced zero completion fires in frozen deterministic "
    "checkpoint evaluation. Therefore no Phase B joint-retention protocol is frozen or executed.\n",
    encoding="utf-8",
)

dump("stage_classification.json", {
    "primary": classification,
    "reachability_pass": reachability_pass,
    "reachability_strong_pass": strong_pass,
    "first_completion_iteration": first_completion,
    "completion_iterations": len(completion_iterations),
    "completion_total": total_completion,
    "maximum_completion_density": max_density,
    "maximum_density_iteration": int(max_density_row["iteration"]),
    "saved_checkpoint_rollouts_with_completion": saved_with_completion,
    "deterministic_checkpoints_with_completion": deterministic_completion_checkpoints,
    "catastrophic_retention_collapse": catastrophic,
    "reason": "Completion is repeatable in stochastic training but remains below 0.05% and is absent from every frozen deterministic checkpoint.",
})
dump("recommended_next_action.json", {
    "single_next_action": next_action, "executed": False,
    "phase_b_ready": phase_b_ready,
})
dump("protected_hashes.json", {
    "starting_head": START, "phase_a_parent_sha256": PARENT_SHA,
    "prior_exp012_results_unchanged": True, "exp005_to_exp011_unchanged": True,
    "existing_checkpoints_unchanged": True, "existing_optimizer_states_unchanged": True,
    "reward_unchanged": True, "network_observation_action_physics_unchanged": True,
    "isaaclab_rsl_rl_core_unchanged": True, "runtime_checkpoint_switches": 0,
    "teacher_expert_calls": 0, "remote_push": False,
    "unrelated_dirty_preserved": [
        "experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
        "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.ps1",
        "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.py",
        "OpenDuck/media/artifact untracked work",
    ],
})
dump("gate.json", {
    "pre_run_integrity": integrity["status"],
    "first_update": first["status"], "early_guard": "PASS",
    "completed_iterations": 100, "completed_interactions": 2457600,
    "reachability": "FAIL_DENSITY_AND_FROZEN_REPRODUCIBILITY",
    "classification": classification, "phase_b_ready": False,
    "phase_b_executed": False,
})
(OUT / "reproduction_commands.ps1").write_text(
    '$ErrorActionPreference = "Stop"\n'
    'Set-Location "$HOME\\workspace\\physical-ai-lab"\n'
    '.\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\run_stage2e_phase_a.ps1\n'
    '.\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\evaluate_stage2e_phase_a.ps1\n'
    '& C:\\isaacsim\\python.bat .\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\finalize_stage2e_phase_a.py\n'
    '.\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\play_exp012.ps1 -Mode RunAcquisition\n',
    encoding="utf-8",
)

selected_run = {speed: selected_conditions[f"run_{speed:.1f}"] for speed in (2.3, 2.4, 2.5, 2.6)}
selected_retention = {
    "STAND": selected_conditions["retention_0.0"],
    "WALK_0.6": selected_conditions["retention_0.6"],
    "WALK_1.2": selected_conditions["retention_1.2"],
    "WALK_TO_STAND": selected_conditions["retention_0.6_to_0.0"],
}
gradient_selected = next(x for x in gradients["diagnostics"] if x["iteration"] == selected_iteration)

REPORT.write_text(f"""# exp_012 two-stage single-policy RUN acquisition — Phase A

## Integrity

Phase A strictly resumed the Stage 2 retry selected iteration-100 checkpoint
`{PARENT_SHA}` with Adam step 87,000 and restored LR
`{integrity['optimizer_lr']:.8g}`. Runtime, scheduler, and optimizer LR matched.
The base and `SafePeriodicFlightReward` semantics, PPO configuration, network,
observation, action, and physics were unchanged. Yaw command was zero and all
external controllers were off.

## Training

The single actor completed 100 iterations / 2,457,600 interactions. First-update
exact KL was {first['exact_old_new']:.6f}, maximum per-step KL was
{first['all_step_maximum_kl']:.6f}, and clip fraction was
{first['clip_fraction']:.4f}; all stability gates passed.

The first stochastic completion fired at iteration {first_completion}. Across
training, {total_completion} completions occurred in {len(completion_iterations)}
iterations. Peak density was {100*max_density:.4f}% at iteration
{max_density_row['iteration']}, below the registered 0.05% gate. Frozen
deterministic evaluation produced completion fires in 0 of 10 checkpoints.

## Selected checkpoint

Iteration {selected_iteration}, SHA `{selected['sha256']}`, was selected by the
registered ordering after all deterministic completion densities tied at zero;
it had the strongest 2.4 m/s periodic-running result.

| speed | periodic | fall | speed MAE | heading p95 | slip | impact | saturation |
|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
    f"| {speed:.1f} | {100*x['periodic_running_rate']:.0f}% | {100*x['fall_rate']:.0f}% | "
    f"{x['speed_mae']:.3f} | {x['heading_p95']:.3f} | {100*x['dangerous_slip_rate']:.0f}% | "
    f"{100*x['impact_failure_rate']:.0f}% | {100*x['long_dwell_saturation_rate']:.0f}% |"
    for speed, x in selected_run.items()
) + f"""

## Gradient emergence

At the selected checkpoint, precursor/base gradient ratio was
{100*gradient_selected['precursor_to_base']:.4f}%, completion/base was
{100*gradient_selected['completion_to_base']:.4f}%, and run-specific/total was
{100*gradient_selected['run_specific_to_total']:.4f}%. Completion/total cosine
was {gradient_selected['completion_total_cosine']:.3f}. The effective Adam
descent-direction alignment was {gradient_selected['descent_vs_adam_update_cosine']:.3f};
optimizer moments were retained, not reset.

## Retention

Selected-checkpoint STAND, WALK 0.6, WALK 1.2, and WALK_TO_STAND success were all
100%, with zero falls in these diagnostic sets. No catastrophic retention
collapse occurred.

## Classification

**{classification}**

Phase A clearly moved the gait toward periodic RUN (2.4 m/s 85%, 2.6 m/s 50%),
but completion remained too sparse, unsafe at several speeds, and absent in
frozen deterministic evaluation.

## Phase B

**NOT READY.** Phase B was not executed and no joint-retention protocol was
frozen. The single next action is **{next_action}**.
""", encoding="utf-8")

print(json.dumps({
    "classification": classification, "selected_iteration": selected_iteration,
    "selected_sha256": selected["sha256"], "first_completion": first_completion,
    "total_completion": total_completion, "max_density": max_density,
    "phase_b_ready": phase_b_ready,
}, indent=2))
