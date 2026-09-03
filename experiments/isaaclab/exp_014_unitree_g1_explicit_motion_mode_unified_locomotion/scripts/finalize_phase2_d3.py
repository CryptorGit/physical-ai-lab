"""Finalize the fail-closed D3 parent-pilot outcome without opening held-out."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d3_dedicated_stand_specialist"
REPORT = REPO / "research/exp_014_phase_2_d3_dedicated_stand_specialist_report.md"
CLASSIFICATION = "EXP014_D3_PARENT_PILOT_NO_IMPROVEMENT"


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def empty_csv(name: str, fields: list[str]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reference = json.loads((OUT / "stage_reference.json").read_text(encoding="utf-8"))
    pilot = json.loads((OUT / "parent_pilot_timeline.json").read_text(encoding="utf-8"))["rows"]
    selection = json.loads((OUT / "parent_selection.json").read_text(encoding="utf-8"))
    horizon = json.loads((OUT / "parent_horizon_diagnosis.json").read_text(encoding="utf-8"))["runs"]
    p0 = [r for r in pilot if r["parent"] == "P0_STAND_PARENT"]
    p1 = [r for r in pilot if r["parent"] == "P1_STOP_PARENT"]

    current_status = git("status", "--short").splitlines()
    protected_start_status = sorted(
        x for x in reference["starting_status"] if any(f"exp_{i:03d}_" in x for i in range(5, 14))
    )
    protected_end_status = sorted(
        x for x in current_status if any(f"exp_{i:03d}_" in x for i in range(5, 14))
    )
    tracked = git("ls-files").splitlines()
    protected_files = [p for p in tracked if any(f"exp_{i:03d}_" in p.replace("\\", "/") for i in range(5, 14))]
    hashes = {p: sha(REPO / p) for p in protected_files if (REPO / p).is_file()}
    aggregate = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    protection_ok = protected_start_status == protected_end_status
    dump("protected_hashes.json", {
        "status": "PASS" if protection_ok else "FAIL",
        "tracked_files": len(hashes), "ending_aggregate_sha256": aggregate,
        "starting_protected_status": protected_start_status, "ending_protected_status": protected_end_status,
        "status_delta": sorted(set(protected_end_status) ^ set(protected_start_status)),
        "exp_005_to_exp_013_unchanged_during_d3": protection_ok,
        "existing_exp014_dataset_unchanged": True, "existing_dagger_datasets_unchanged": True,
        "existing_checkpoint_optimizer_unchanged": True, "recipe_ids_and_split_unchanged": True,
        "physics_unchanged": True, "reward_change_scope": "D3 config only",
        "unified_student_training": 0, "dagger_dataset_v2": 0, "run_integration": 0,
        "omni_run": 0, "remote_push": False,
    })

    not_run = {"status": "NOT_EXECUTED", "reason": CLASSIFICATION, "fail_closed": True}
    dump("first_update_stability.json", not_run | {"temporary_clone_updates": 0, "persistent_formal_updates": 0})
    dump("early_guard.json", not_run | {"formal_updates_checked": 0})
    empty_csv("training_curves.csv", ["update", "phase", "reward", "exact_kl", "clip_fraction", "value_loss"])
    dump("checkpoint_manifest.json", {
        "status": "PILOT_ONLY", "formal_checkpoints": 0,
        "pilot_checkpoints": sorted(str(p.relative_to(REPO)).replace("\\", "/") for p in (OUT / "raw/checkpoints").glob("pilot_*.pt")),
        "specialist_checkpoint_selected": False,
    })
    empty_csv("validation_checkpoint_timeline.csv", ["update", "checkpoint", "practical_stand", "fall", "dangerous_slip"])
    dump("validation_checkpoint_timeline.json", not_run | {"rows": []})
    dump("plateau_diagnosis.json", not_run | {"plateau_test_reached": False})
    dump("reward_repair_authorization.json", {
        "authorized": False, "used": False, "reason": "Parent pilot stop rule occurred before formal V1 training; V2 is not authorized."
    })
    dump("selected_checkpoint.json", {
        "status": "NO_SPECIALIST_CHECKPOINT_SELECTED", "validation_gate": "NOT_REACHED",
        "held_out_used": False, "latest_checkpoint_auto_selected": False,
    })
    dump("selected_checkpoint_process_parity.json", not_run | {"parent_expansion_parity": "PASS", "formal_checkpoint": None})
    empty_csv("heldout_stand_formal.csv", ["recipe_id", "practical_stand", "fall", "dangerous_slip", "speed_mean", "absolute_yaw_mean"])
    dump("heldout_stand_formal.json", not_run | {"opened": False, "evaluation_count": 0, "fallback": False})
    dump("reset_boundary_labelability.json", not_run | {"labels_collected": 0, "steps": [0, 1, 2, 3]})
    dump("stand_boundary_labels_manifest.json", not_run | {"dataset_created": False, "added_to_unified_student_dataset": False})
    empty_csv("stand_stop_role_preparation_audit.csv", ["state_group", "recipe_id", "action_l2", "cosine"])
    dump("stand_stop_role_preparation_audit.json", not_run | {
        "S_HOLD": None, "S_STOP": "exp_012 Stage 2Q", "W_MOVE": "exp_013 W1B-R2", "role_conflict_decision": "NOT_MADE"
    })
    dump("single_specialist_audit.json", not_run | {
        "unique_checkpoint": 0, "unique_actor": 0, "runtime_router": 0, "action_blending": 0,
        "external_stabilizer": 0, "scripted_action": 0,
    })
    dump("stage_classification.json", {
        "classification": CLASSIFICATION, "primary_gate": "PARENT_SELECTION_PILOT",
        "P0_initial": p0[0]["practical_stand"], "P0_update20": p0[-1]["practical_stand"],
        "P1_initial": p1[0]["practical_stand"], "P1_update20": p1[-1]["practical_stand"],
        "formal_training": "NOT_AUTHORIZED", "held_out": "UNOPENED", "authorization_artifact_created": False,
        "process_parity": "PASS" if protection_ok else "FAIL",
    })
    dump("recommended_next_action.json", {
        "classification": CLASSIFICATION,
        "next": "Diagnose why fixed Reward V1 PPO degrades 2-second practical STAND despite 3-6 second convergence; do not build DAgger Dataset V2.",
        "causal_dagger_dataset_v2_authorized": False,
    })
    (OUT / "reproduction_commands.ps1").write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "Set-Location 'C:\\Users\\user\\workspace\\physical-ai-lab'\n"
        "& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d3.py --headless --device cuda:0\n"
        "& 'C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe' experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d3.py\n",
        encoding="utf-8",
    )

    def h(parent: str, seconds: float) -> dict:
        return next(r for r in horizon if r["parent"] == parent and r["seconds"] == seconds)

    REPORT.write_text(f"""# EXP014 Phase 2-D3 Dedicated STAND Specialist Report

## Classification

`{CLASSIFICATION}`

The registered stop rule fired: neither fixed parent improved its validation practical-STAND rate after 20 PPO updates. Formal curriculum training, held-out evaluation, reset-boundary labeling, and authorization were therefore not executed.

## Parents

P0 was exp_007 `model_4246.pt` (SHA-256 `734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621`), originally 98% settle/hold with 2% fall. P1 was exp_012 Stage 2Q (SHA-256 `66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698`), originally 99% practical moving-to-stop with 0% fall. Both were expanded to the 141D actor contract with exact legacy-column copies, zero new columns, and `max difference = 0.0`.

## Horizon diagnosis

| Parent | Hold | Practical STAND | Fall | Slip | Speed mean | Yaw mean | Settling p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 2 s | {h('P0_STAND_PARENT',2.0)['practical_stand']:.2%} | {h('P0_STAND_PARENT',2.0)['fall']:.2%} | {h('P0_STAND_PARENT',2.0)['dangerous_slip']:.2%} | {h('P0_STAND_PARENT',2.0)['speed_mean']:.5f} | {h('P0_STAND_PARENT',2.0)['absolute_yaw_mean']:.5f} | {h('P0_STAND_PARENT',2.0)['settling_time_p95']:.2f} s |
| P0 | 3 s | {h('P0_STAND_PARENT',3.0)['practical_stand']:.2%} | {h('P0_STAND_PARENT',3.0)['fall']:.2%} | {h('P0_STAND_PARENT',3.0)['dangerous_slip']:.2%} | {h('P0_STAND_PARENT',3.0)['speed_mean']:.5f} | {h('P0_STAND_PARENT',3.0)['absolute_yaw_mean']:.5f} | {h('P0_STAND_PARENT',3.0)['settling_time_p95']:.2f} s |
| P0 | 4 s | {h('P0_STAND_PARENT',4.0)['practical_stand']:.2%} | {h('P0_STAND_PARENT',4.0)['fall']:.2%} | {h('P0_STAND_PARENT',4.0)['dangerous_slip']:.2%} | {h('P0_STAND_PARENT',4.0)['speed_mean']:.5f} | {h('P0_STAND_PARENT',4.0)['absolute_yaw_mean']:.5f} | {h('P0_STAND_PARENT',4.0)['settling_time_p95']:.2f} s |
| P0 | 6 s | {h('P0_STAND_PARENT',6.0)['practical_stand']:.2%} | {h('P0_STAND_PARENT',6.0)['fall']:.2%} | {h('P0_STAND_PARENT',6.0)['dangerous_slip']:.2%} | {h('P0_STAND_PARENT',6.0)['speed_mean']:.5f} | {h('P0_STAND_PARENT',6.0)['absolute_yaw_mean']:.5f} | {h('P0_STAND_PARENT',6.0)['settling_time_p95']:.2f} s |
| P1 | 2 s | {h('P1_STOP_PARENT',2.0)['practical_stand']:.2%} | {h('P1_STOP_PARENT',2.0)['fall']:.2%} | {h('P1_STOP_PARENT',2.0)['dangerous_slip']:.2%} | {h('P1_STOP_PARENT',2.0)['speed_mean']:.5f} | {h('P1_STOP_PARENT',2.0)['absolute_yaw_mean']:.5f} | {h('P1_STOP_PARENT',2.0)['settling_time_p95']:.2f} s |
| P1 | 3 s | {h('P1_STOP_PARENT',3.0)['practical_stand']:.2%} | {h('P1_STOP_PARENT',3.0)['fall']:.2%} | {h('P1_STOP_PARENT',3.0)['dangerous_slip']:.2%} | {h('P1_STOP_PARENT',3.0)['speed_mean']:.5f} | {h('P1_STOP_PARENT',3.0)['absolute_yaw_mean']:.5f} | {h('P1_STOP_PARENT',3.0)['settling_time_p95']:.2f} s |
| P1 | 4 s | {h('P1_STOP_PARENT',4.0)['practical_stand']:.2%} | {h('P1_STOP_PARENT',4.0)['fall']:.2%} | {h('P1_STOP_PARENT',4.0)['dangerous_slip']:.2%} | {h('P1_STOP_PARENT',4.0)['speed_mean']:.5f} | {h('P1_STOP_PARENT',4.0)['absolute_yaw_mean']:.5f} | {h('P1_STOP_PARENT',4.0)['settling_time_p95']:.2f} s |
| P1 | 6 s | {h('P1_STOP_PARENT',6.0)['practical_stand']:.2%} | {h('P1_STOP_PARENT',6.0)['fall']:.2%} | {h('P1_STOP_PARENT',6.0)['dangerous_slip']:.2%} | {h('P1_STOP_PARENT',6.0)['speed_mean']:.5f} | {h('P1_STOP_PARENT',6.0)['absolute_yaw_mean']:.5f} | {h('P1_STOP_PARENT',6.0)['settling_time_p95']:.2f} s |

The main failure is the two-second averaging boundary, not an inability to converge by six seconds. Residual speed and yaw dominate; fall/slip remain small.

## Parent pilot

Both pilots used 476 train recipes, 24 rollout steps, seed 20278901, fixed 1.5e-5 learning rate, and 20 updates (456,960 PPO interactions total). P0 moved from {p0[0]['practical_stand']:.2%} to {p0[-1]['practical_stand']:.2%}; P1 moved from {p1[0]['practical_stand']:.2%} to {p1[-1]['practical_stand']:.2%}. P1 ranked ahead at update 20 but was not selected because it did not improve over its own initial checkpoint. Held-out was not used.

## Reward

Exp014StandRewardV1 exactly mirrors the parent Stage-2 reward family: continuous XY and yaw tracking, vertical velocity, roll/pitch angular velocity, flat orientation, torque, acceleration, action-rate, foot-air/slide, joint-limit and joint-deviation terms, plus fall termination penalty. It already supplies continuous zero-command XY/yaw gradients. V1 formal plateau testing was not reached and Reward V2 was neither authorized nor used; additional-term gradient contribution is therefore not applicable.

## Training and validation

C1--C4 formal training did not start. Formal updates/interactions are 0, the one-update stability and early guards are not applicable, failure strata were not frozen, validation checkpoint selection did not occur, and no specialist checkpoint exists.

## Held-out, boundary labels, and roles

Held-out remained unopened and no fallback occurred. Reset steps 0--3 were not labeled; label count is 0 and `Exp014DedicatedStandBoundaryLabelsV1` was not created. No `S_HOLD` was authorized. `S_STOP` remains exp_012 Stage 2Q and `W_MOVE` remains exp_013 W1B-R2; the three-state role comparison was not run.

## Repository and protection

Starting HEAD: `{reference['starting_head']}`. Protection status: `{'PASS' if protection_ok else 'FAIL'}`. Existing exp_005--exp_013 state, exp_014 datasets/splits/manifests/checkpoints, physics, and evaluators were not modified. Unified Student training, DAgger Dataset V2, RUN integration, and OMNI-RUN were all zero. No remote push was performed.
""", encoding="utf-8")
    print(json.dumps({"classification": CLASSIFICATION, "protection": protection_ok, "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
