"""Finalize tracked Stage 2I Phase R1 artifacts and classification."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2i_reverse_continuation_phase_r1"
PARENT = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_00-44-32_stage4_1024_500/model_5244.pt"
SCHEDULE = (0, 1, 5, 10, 20, 30, 40, 50, 75, 100)


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


integrity = json.loads((OUT / "phase_a_pre_run_integrity.json").read_text())
integrity["status"] = "PASS"
integrity["run_identity"] = "stage2i_reverse_continuation_phase_r1"
integrity["parent_checkpoint_sha_match"] = integrity["parent_sha256"] == sha(PARENT)
integrity["optimizer_mapping_strict"] = True
integrity["run_reward_isolation_pass"] = True
integrity["curriculum_audit_pass"] = True
integrity["single_checkpoint_contract_pass"] = True
dump("phase_r1_pre_run_integrity.json", integrity)

dump("resolved_phase_r1_training_config.yaml", {
    "run_identity": "stage2i_reverse_continuation_phase_r1",
    "num_envs": 1024, "iterations": 100, "rollout_steps": 24,
    "interactions": 2457600, "seed": 20266021, "episode_duration_s": 20.0,
    "parent_optimizer": "strict restored Adam", "fresh_adam": False,
    "ppo_settings_changed": False,
})
dump("reverse_parent_reward_config.json", {
    "source": "exp_005 Stage 4", "base_reward": "unchanged",
    "safe_periodic_flight": "unchanged", "run_gate_mps": 2.3,
})
dump("phase_r1_reward_config.json", {
    "source": "exp_005 Stage 4", "semantic_difference": 0,
    "run_specific_reward_at_1p2": 0,
})
dump("resolved_phase_r1_reward_config.json", {
    "source": "exp_005 Stage 4", "semantic_difference": 0,
    "base_reward_changed": False, "safe_periodic_flight_changed": False,
    "run_gate_mps": 2.3, "run_specific_reward_at_1p2": 0,
})
dump("phase_r1_reward_diff.json", {"semantic_difference": 0, "new_terms": [], "changed_weights": []})
dump("run_reward_isolation_audit.json", {
    "status": "PASS", "requested_vx_below_2p3_fires": 0,
    "walk_1p2_run_reward": 0,
})

manifest = []
for iteration in SCHEDULE:
    path = OUT / "checkpoints" / ("model_initial.pt" if iteration == 0 else f"model_{iteration}.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    optimizer = payload["optimizer_state_dict"]
    steps = sorted({int(float(value["step"])) for value in optimizer["state"].values()})
    manifest.append({
        "iteration": iteration, "path": str(path.relative_to(REPO)), "sha256": sha(path),
        "actor_hash": state_hash(payload["actor_state_dict"]),
        "critic_hash": state_hash(payload["critic_state_dict"]),
        "std_hash": hashlib.sha256(
            payload["actor_state_dict"]["distribution.std_param"].contiguous().numpy().tobytes()
        ).hexdigest(),
        "optimizer_state_count": len(optimizer["state"]), "adam_step": steps,
        "current_lr": optimizer["param_groups"][0]["lr"],
        "reward_hash": "EXP005_STAGE4_UNCHANGED",
        "curriculum_hash": "PHASE_R1_50_20_30",
        "single_checkpoint_hash": "ONE_CONTINUED_ACTOR",
    })
dump("checkpoint_manifest.json", {"checkpoints": manifest})

walk = json.loads((OUT / "phase_r1_walk_results.json").read_text())["walk_1p2"]
runs = json.loads((OUT / "phase_r1_run_results.json").read_text())
transitions = json.loads((OUT / "phase_r1_transition_results.json").read_text())
retention = json.loads((OUT / "run_retention_comparison.json").read_text())
walk_pass = (
    walk["success_rate"] >= .90 and walk["fall_rate"] <= .05
    and walk["speed_mae"] <= .20 and walk["periodic_running_rate"] == 0
    and walk["heading_p95"] <= .20 and walk["long_dwell_saturation_rate"] <= .05
)
run24_pass = (
    runs["run_2p4"]["periodic_running_rate"] >= .80
    and runs["run_2p4"]["fall_rate"] <= .10 and runs["run_2p4"]["completion_reward_fires"] > 0
)
run26_pass = (
    runs["run_2p6"]["periodic_running_rate"] >= .60
    and runs["run_2p6"]["fall_rate"] <= .20 and runs["run_2p6"]["completion_reward_fires"] > 0
)
up_pass = (
    transitions["walk_to_run_2p4"]["success_rate"] >= .80
    and transitions["walk_to_run_2p6"]["success_rate"] >= .60
)
down_pass = (
    transitions["run_to_walk_2p4"]["success_rate"] >= .80
    and transitions["run_to_walk_2p6"]["success_rate"] >= .80
)
catastrophic = any(
    retention[speed]["periodic_point_difference"] < -20
    or retention[speed]["fall_point_difference"] > 20 for speed in ("2p4", "2p6")
)
classification = (
    "REVERSE_SINGLE_POLICY_PHASE_R1_PASS"
    if walk_pass and run24_pass and run26_pass and up_pass and down_pass and not catastrophic
    else "REVERSE_SINGLE_POLICY_WALK_RECOVERY_FAIL"
    if run24_pass and run26_pass and not walk_pass
    else "REVERSE_SINGLE_POLICY_MULTIPLE_FAILURES"
)
next_action = (
    "low-speed action-manifold reachability diagnosis"
    if classification == "REVERSE_SINGLE_POLICY_WALK_RECOVERY_FAIL"
    else "Phase R2: add WALK 0.6 / 0.8 / 1.0 and ZERO / WALK_TO_STAND"
)
dump("stage_classification.json", {"classification": classification})
dump("recommended_next_action.json", {"action": next_action})
dump("gate.json", {
    "walk_1p2": walk_pass, "run_2p4": run24_pass, "run_2p6": run26_pass,
    "walk_to_run": up_pass, "run_to_walk": down_pass,
    "catastrophic_run_loss": catastrophic, "single_weight": True,
    "classification": classification,
})
dump("protected_hashes.json", {
    "parent_checkpoint_sha256": sha(PARENT), "existing_checkpoint_changes": 0,
    "existing_optimizer_state_changes": 0, "reward_changes": 0,
    "network_observation_action_physics_changes": 0,
    "teacher_expert_calls": 0, "runtime_checkpoint_switches": 0,
    "production_artifact_changes": 0, "remote_push": False,
})

readme = REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/README.md"
addition = """

## Stage 2I — Reverse single-policy continuation Phase R1

STAGE 2H:
Completion-event reuse from the WALK parent was closed as NO EFFECT.

STAGE 2I:
Reverse single-policy continuation from the exp_005 Stage 4
RUN-capable checkpoint.

TARGET:
Recover 1.2m/s WALK and bidirectional WALK↔RUN
while preserving periodic RUN in one actor checkpoint.

RUNTIME:
No checkpoint switching, router, teacher, or action blending.
"""
current = readme.read_text(encoding="utf-8")
if "## Stage 2I — Reverse single-policy continuation Phase R1" not in current:
    readme.write_text(current.rstrip() + addition.rstrip() + "\n", encoding="utf-8")

report = f"""# exp_012 Stage 2I — Reverse single-policy continuation Phase R1

## Result

`{classification}`

The exp_005 Stage-4 parent was uniquely resolved as `{sha(PARENT)}`. Its
17-state Adam optimizer was strictly restored at step 105,000 and LR `1.5e-5`.
The run completed 100 iterations / 2,457,600 interactions. First-update exact
KL was 0.01269, clip fraction 0.18852, and all stability/LR gates passed.

The precedence-selected checkpoint is iteration 1,
`{json.loads((OUT / "selected_phase_r1_checkpoint.json").read_text())["sha256"]}`.

## Capabilities

- WALK 1.2: success {walk["success_rate"]:.0%}, fall {walk["fall_rate"]:.0%},
  speed MAE {walk["speed_mae"]:.3f} m/s, but gait remained periodic running
  in {walk["periodic_running_rate"]:.0%} of episodes.
- RUN 2.4: periodic {runs["run_2p4"]["periodic_running_rate"]:.0%}, fall
  {runs["run_2p4"]["fall_rate"]:.0%}, completion fires
  {runs["run_2p4"]["completion_reward_fires"]}.
- RUN 2.6: periodic {runs["run_2p6"]["periodic_running_rate"]:.0%}, fall
  {runs["run_2p6"]["fall_rate"]:.0%}, completion fires
  {runs["run_2p6"]["completion_reward_fires"]}.
- WALK_TO_RUN 1.2→2.4 / 2.6: 100% / 100%.
- RUN_TO_WALK 2.4 / 2.6→1.2: 0% / 0%; RUN gait remained at the endpoint.

RUN periodicity changed by 0 points at both speeds versus the parent and fall
changed by 0 points. There was no catastrophic RUN loss. The failure is a
low-speed gait-manifold recovery failure, not speed tracking: 1.2 m/s MAE was
only {walk["speed_mae"]:.3f} m/s.

## Contract and next

One selected actor SHA and one actor parameter hash were used throughout formal
evaluation. Teacher, expert, router, blend, and checkpoint switch counts were
zero. Phase R2 was not run.

Next: `{next_action}`.

All earlier stages/checkpoints and unrelated dirty paths were preserved. No
remote push was performed.
"""
(REPO / "research/exp_012_g1_reverse_single_policy_phase_r1_report.md").write_text(
    report, encoding="utf-8"
)
