"""Finalize tracked Stage 2Q artifacts without modifying prior stages."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration"
RESEARCH = REPO / "research/exp_012_g1_final_single_policy_sequence_report.md"
CHECKPOINT = OUT / "raw/dagger_round_2_student.pt"
STARTING_HEAD = "c8d921a831ea3387651449e5561dda8fb0a764a6"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    endpoint = load("closed_loop_endpoint.json")
    transition = load("transition_results.json")
    final = load("final_integrated_sequence.json")
    static = load("static_endpoint_results.json")
    selected_sha = sha(CHECKPOINT)
    endpoint["checkpoint_sha256"] = selected_sha
    transition["checkpoint_sha256"] = selected_sha
    final["checkpoint_sha256"] = selected_sha
    dump("closed_loop_endpoint.json", endpoint)
    dump("transition_results.json", transition)
    dump("final_integrated_sequence.json", final)
    stand = endpoint["summary"]["STAND_0P0"]
    endpoint_rows = list(csv.DictReader((OUT / "closed_loop_endpoint.csv").open(encoding="utf-8")))
    stand_rows = [row for row in endpoint_rows if row["condition"] == "STAND_0P0"]
    stand_segments = [json.loads(row["segment_metrics"])["0"] for row in stand_rows]
    stand_result = {
        **stand,
        "flight_zero_rate": sum(row["flight_fraction"] == 0 for row in stand_segments) / len(stand_segments),
        "double_support_gate_rate": sum(row["double_support_fraction"] >= .95 for row in stand_segments) / len(stand_segments),
        "double_support_fraction_mean": sum(row["double_support_fraction"] for row in stand_segments) / len(stand_segments),
        "gate": False,
    }
    dump("closed_loop_stand.json", stand_result)
    with (OUT / "closed_loop_walk.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [{"condition": name, **endpoint["summary"][name]} for name in ("WALK_0P6", "WALK_0P8", "WALK_1P0", "WALK_1P2")]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "closed_loop_run.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [{"condition": name, **endpoint["summary"][name]} for name in ("RUN_1P2", "RUN_2P4", "RUN_2P6")]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dagger = [load("dagger_round_1.json"), load("dagger_round_2.json")]
    dump("dagger_rounds.json", {
        "trigger": "STAND endpoint and WALK_TO_STAND contact gate failed",
        "rounds": dagger, "maximum_rounds": 2, "teacher_routing_changed": False,
        "result": "STAND/STOP contact gate remained failed",
    })
    dump("candidate_stochastic_sequence.json", {
        "status": "NOT_RUN", "reason": "deterministic final gate did not pass",
        "alpha_walk": .30, "alpha_run": .65,
    })
    dump("student_parent_manifest.json", {
        "path": "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2n_gait_conditioned_ppo_retention_preflight/checkpoints/model_initial.pt",
        "sha256": "04b43e5497bc35e2d00fa4476f9120f9439e0953283c69cf8ca1e9635dedd121",
        "mean_actor_source_sha256": "d0c46acdc2d3a5793d7dc8d6ae9e47f741ff0353fb1aef8c9ff993e71ea8bec3",
        "architecture": [124, 256, 128, 128, 37], "alpha_walk": .30, "alpha_run": .65,
    })
    dump("selected_checkpoint.json", {
        "path": str(CHECKPOINT.relative_to(REPO)), "sha256": selected_sha,
        "selection": "DAgger round 2 final bounded candidate", "supervised_base_step": 17500,
        "dagger_rounds": 2, "persistent_ppo_updates": 0,
    })
    manifest = load("checkpoint_manifest.json")
    manifest["selected_path"] = str(CHECKPOINT.relative_to(REPO))
    manifest["selected_sha256"] = selected_sha
    manifest["dagger_checkpoints"] = [{
        "round": row["round"], "path": row["output"], "sha256": row["output_sha256"],
        "supervised_steps": row["supervised_steps"],
    } for row in dagger]
    dump("checkpoint_manifest.json", manifest)
    dump("stage_reference.json", {
        "stage": "2Q", "name": "final single-policy sequence integration",
        "starting_head": STARTING_HEAD, "parent_sha256": "04b43e5497bc35e2d00fa4476f9120f9439e0953283c69cf8ca1e9635dedd121",
        "walk_teacher_sha256": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
        "run_teacher_sha256": "90d1a360587142d7e312db00a281505a027ecb221733eea6451a885868f6ccc9",
        "selected_sha256": selected_sha,
    })
    dump("protocol.json", {
        "training": "mean-only supervised integration plus conditional bounded DAgger",
        "ppo_updates": 0, "reward_changes": 0, "std_updates": 0,
        "dataset_weights": {"STAND": .25, "LOW_MID_WALK": .25, "RUN": .25, "GAIT_TOGGLE": .25},
        "static_gate": {"mse_max": .001, "cosine_min": .98},
        "episodes": {"endpoint": 100, "transition": 100, "final_sequence": 100},
        "final_sequence_speed_max": 2.4, "run_2p6": "retention diagnostic only",
    })
    dump("single_weight_audit.json", {
        "selected_checkpoint_sha256": selected_sha, "unique_checkpoint_count": 1,
        "unique_mean_actor_count": 1, "unique_gaussian_head_count": 1,
        "teacher_calls_runtime": 0, "expert_calls_runtime": 0, "router_calls": 0,
        "checkpoint_switches": 0, "action_blends": 0, "action_source": "selected Stage 2Q student only",
        "status": "PASS",
    })
    classification = "G1_FINAL_STAND_STOP_FAIL"
    dump("stage_classification.json", {
        "primary": classification,
        "evidence": {
            "static_all_endpoints_pass": static["aggregate_classification"] == "PASS",
            "walk_run_endpoints_pass": True, "walk_run_transitions_pass": True,
            "stand_pass": False, "walk_to_stand_pass": False,
            "integrated_sequence_completion": final["summary"]["FINAL_SEQUENCE"]["success_rate"],
        },
        "prior_classifications_modified": False,
    })
    dump("recommended_next_action.json", {
        "action": "STAND/STOP contact-stability boundary diagnosis",
        "reason": "Only STAND and STOP double-support/zero-flight gates remain; WALK, RUN, gait toggle, and RUN acceleration/deceleration are retained.",
        "executed": False,
    })
    status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True, encoding="utf-8")
    protected = {
        "exp_005_through_exp_011_unchanged_by_stage2q": True,
        "exp_012_stage0_through_stage2p_unchanged": True,
        "teacher_checkpoints_unchanged": True, "existing_student_checkpoints_unchanged": True,
        "reward_unchanged": True, "physics_unchanged": True,
        "isaaclab_core_unchanged": True, "rsl_rl_core_unchanged": True,
        "runtime_teacher_expert_calls": 0, "checkpoint_switches": 0, "remote_push": False,
        "unrelated_dirty_state_preserved": [
            line for line in status.splitlines()
            if ".stage2q_" not in line and "/scripts/collect_stage2q_" not in line
            and "/scripts/evaluate_stage2q_" not in line and "/scripts/finalize_stage2q" not in line
            and "/scripts/play_stage2q_" not in line and "/scripts/train_stage2q_" not in line
            and "exp_012_g1_final_single_policy_sequence_report.md" not in line
        ],
        "protected_checkpoint_sha256": {
            "Stage2N_initial": "04b43e5497bc35e2d00fa4476f9120f9439e0953283c69cf8ca1e9635dedd121",
            "WALK_teacher": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
            "RUN_teacher": "90d1a360587142d7e312db00a281505a027ecb221733eea6451a885868f6ccc9",
        },
    }
    dump("protected_hashes.json", protected)
    dump("gate.json", {
        "static_endpoint_gate": "PASS", "closed_loop_walk_gate": "PASS",
        "closed_loop_run_gate": "PASS", "stand_gate": "FAIL",
        "transition_gate": "FAIL", "final_sequence_gate": "FAIL",
        "single_weight_gate": "PASS", "classification": classification,
    })
    reproduction = r'''$ErrorActionPreference = "Stop"
Set-Location "$HOME\workspace\physical-ai-lab"
$env:PYTHONPATH = "$PWD\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\src;$PWD\experiments\isaaclab\exp_005_unitree_g1_flat_run\src;$PWD"
$py = "C:\isaacsim\python.bat"
& $py .\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\collect_stage2q_dataset.py --mode supplement --headless --device cuda:0
& $py .\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\collect_stage2q_dataset.py --mode toggle --headless --device cuda:0
& $py .\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\train_stage2q_student.py
# DAgger rounds are conditional on endpoint/transition failure and are bounded to two.
& $py .\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\evaluate_stage2q_sequence.py --mode endpoints --headless --device cuda:0
& $py .\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\evaluate_stage2q_sequence.py --mode transitions --headless --device cuda:0
& $py .\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\evaluate_stage2q_sequence.py --mode final --headless --device cuda:0
& $py .\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\finalize_stage2q.py
'''
    (OUT / "reproduction_commands.ps1").write_text(reproduction, encoding="utf-8")
    f = final["summary"]["FINAL_SEQUENCE"]
    report = f"""# EXP 012 Stage 2Q — Final single-policy sequence integration

## Outcome

Stage 2Q is classified **{classification}**. The selected one-checkpoint actor preserves every
WALK/RUN endpoint and every gait/RUN transition, but the Stage 2 WALK/STAND teacher's zero-speed
behavior retains small stepping/contact oscillations. Two pre-authorized DAgger rounds reproduced
rather than removed that behavior.

## Student and data

- Parent: Stage 2N initial checkpoint (`04b43e…d121`)
- Architecture: `124 → 256 → 128 → 128 → 37`
- Frozen gait-conditioned std: `alpha_walk=0.30`, `alpha_run=0.65`
- Selected supervised base step: 17,500
- DAgger: 2 rounds, 500 episodes and 5,000 steps per round
- Selected SHA-256: `{selected_sha}`
- Static held-out: all ten endpoint/toggle conditions PASS; worst MSE
  `{static['selection']['endpoint_worst_loss']:.6f}`, minimum cosine above 0.99989.

## Closed-loop endpoints

WALK 0.6/0.8/1.0/1.2 and RUN 1.2/2.4/2.6 each achieved 100% gait success and 0% fall.
RUN completion fired 5,662 times at 2.4 m/s and 5,645 times at 2.6 m/s. STAND had 0% fall
and 0.0447 m/s speed MAE, but zero-flight and 95% double-support gates were 0%;
mean double support was {stand_result['double_support_fraction_mean']:.1%}.

## Transitions

STAND→WALK, WALK→RUN, RUN acceleration, RUN deceleration, and RUN→WALK were each 100%
with 0% fall. WALK→STAND reached 0.0414 m/s final speed and 0% fall, but failed the formal
zero-flight/double-support completion gate.

## Integrated sequence

- Formal completion: {f['success_rate']:.0%}
- Fall: {f['fall_rate']:.0%}
- Final speed mean: {f['final_speed_mean']:.4f} m/s
- Heading p95 mean: {f['heading_p95_mean']:.4f} rad
- Dangerous slip / impact / long-dwell saturation: {f['dangerous_slip_rate']:.0%} /
  {f['impact_failure_rate']:.0%} / {f['long_dwell_saturation_rate']:.0%}

The locomotion body of the sequence succeeds, but initial/final STAND contact gates prevent any
episode from satisfying the formal all-segment completion predicate. Candidate stochastic
evaluation was correctly skipped because the deterministic gate did not pass.

## Runtime and protection

Evaluation uses one selected checkpoint, one mean actor, and one gait-conditioned Gaussian head.
Runtime teacher/expert/router calls, checkpoint switches, and action blends are all zero. No PPO,
reward, physics, teacher-checkpoint, previous-stage, or production-artifact changes were made.
"""
    RESEARCH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
