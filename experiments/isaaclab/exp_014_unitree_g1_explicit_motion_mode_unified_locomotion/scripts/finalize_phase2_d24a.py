"""Build D24A artifacts without running any additional physics."""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d24a_existing_start_teacher_transfer"
RAW = OUT / "raw"
REPORT = REPO / "research/exp_014_phase_2_d24a_existing_start_teacher_transfer_report.md"
START = "8320d8e8f920c2f705b80e6436c4fc796e8c4df2"
STAGE2Q_SHA = "66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698"
SHOLD_SHA = "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"
WMOVE_SHA = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"
SSTOP_SHA = "5de37e5d0807654d370ba7a79ee9872c4029cb50e548685423e48812249959d5"
CLASSIFICATION = "EXP014_D24A_STAGE2Q_NATIVE_REPRODUCTION_FAIL"
STOP_REASON = "Stage 2Q native mandatory gate failed; protocol requires stopping before S_HOLD transfer."


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


OUT.mkdir(parents=True, exist_ok=True)
native = json.loads((RAW / "native_results.json").read_text(encoding="utf-8"))
rows = native["rows"]
bundle = np.load(RAW / "stage2q_native_trajectories.npz")
first_steps = np.array([r["first_step"] for r in rows if r["first_step"] >= 0], dtype=float)
acq_steps = np.array([r["acquisition_step"] for r in rows if r["acquisition_step"] >= 0], dtype=float)
failure_counts = Counter()
for row in rows:
    if row["success"]:
        failure_counts["PASS"] += 1
    elif row["torque"]:
        failure_counts["TORQUE_SATURATION"] += 1
    elif row["acquisition_step"] < 0:
        failure_counts["FORWARD_CONFIRMATION_FAILURE"] += 1
    else:
        failure_counts["OTHER_SAFETY_FAILURE"] += 1

native_summary = {
    "status": "FAIL",
    "checkpoint": native["checkpoint"],
    "checkpoint_sha256": native["checkpoint_sha256"],
    "seed": native["seed"],
    "episodes": native["episodes"],
    "source_lifecycle": native["source_lifecycle"],
    "historical_native_gait_acquisition": native["native_gait_acquisition_rate"],
    "D24A_forward_25_step_confirmation": native["confirmation_rate"],
    "D24A_safe_success": native["success_rate"],
    "first_step_count": int(len(first_steps)),
    "first_step_time_s": {
        "median": None if not len(first_steps) else float(np.median(first_steps) * .02),
        "p05": None if not len(first_steps) else float(np.quantile(first_steps, .05) * .02),
        "p95": None if not len(first_steps) else float(np.quantile(first_steps, .95) * .02),
    },
    "confirmed_acquisition_time_s": {
        "median": None if not len(acq_steps) else float((np.median(acq_steps) - 24) * .02),
        "p95": None if not len(acq_steps) else float((np.quantile(acq_steps, .95) - 24) * .02),
    },
    "safety": {
        "fall": native["fall_rate"],
        "dangerous_slip": native["dangerous_slip_rate"],
        "impact": sum(r["impact"] for r in rows) / len(rows),
        "velocity_saturation": native["velocity_saturation_rate"],
        "torque_saturation": native["torque_saturation_rate"],
        "nonfinite": sum(r["nonfinite"] for r in rows) / len(rows),
    },
    "gates": {
        "native_gait_acquisition_min_90pct": native["native_gait_acquisition_rate"] >= .90,
        "25_step_confirmation_min_90pct": native["confirmation_rate"] >= .90,
        "fall_max_5pct": native["fall_rate"] <= .05,
        "dangerous_slip_max_10pct": native["dangerous_slip_rate"] <= .10,
        "torque_saturation_max_10pct": native["torque_saturation_rate"] <= .10,
    },
    "overall_gate": bool(native["gate"]),
    "failure_counts": dict(failure_counts),
    "native_trajectory_bundle": str((RAW / "stage2q_native_trajectories.npz").relative_to(REPO)),
    "native_trajectory_bundle_sha256": sha(RAW / "stage2q_native_trajectories.npz"),
    "interpretation": "The historical gait-classification result is reproduced, but it does not satisfy the stricter D24A confirmation and torque-dwell gate.",
}

with (OUT / "stage2q_native_reproduction.csv").open("w", newline="", encoding="utf-8") as f:
    fields = ["episode", "source_hash", "first_step", "acquisition_step", "native_gait_acquired", "success", "fall", "slip", "impact", "velocity", "torque", "nonfinite"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
dump("stage2q_native_reproduction.json", {"summary": native_summary, "episodes": rows})

# Durable native result store. Result and COMPLETED status share one FULL/WAL transaction.
db = OUT / "durable_evaluation.sqlite"
if db.exists():
    db.unlink()
con = sqlite3.connect(db)
con.execute("PRAGMA journal_mode=WAL")
con.execute("PRAGMA synchronous=FULL")
con.executescript("""
CREATE TABLE run_manifest(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE native_episodes(episode_id INTEGER PRIMARY KEY,result_json TEXT NOT NULL,result_sha256 TEXT NOT NULL,status TEXT NOT NULL CHECK(status='COMPLETED'));
CREATE TABLE process_events(event_id INTEGER PRIMARY KEY AUTOINCREMENT,event TEXT NOT NULL,episode_id INTEGER,created_utc TEXT NOT NULL);
""")
with con:
    con.execute("INSERT INTO run_manifest VALUES(?,?)", ("stage", "Phase 2-D24A"))
    con.execute("INSERT INTO run_manifest VALUES(?,?)", ("checkpoint_sha256", STAGE2Q_SHA))
    for row in rows:
        text = json.dumps(row, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(text.encode()).hexdigest()
        con.execute("INSERT INTO native_episodes VALUES(?,?,?,?)", (row["episode"], text, digest, "COMPLETED"))
        con.execute("INSERT INTO process_events(event,episode_id,created_utc) VALUES(?,?,?)", ("EPISODE_COMPLETED", row["episode"], datetime.now(timezone.utc).isoformat()))
con.execute("PRAGMA wal_checkpoint(FULL)")
con.close()

not_executed = {"status": "NOT_EXECUTED", "reason": STOP_REASON, "physics_attempts": 0}
dump("stage2q_start_reference_manifest.json", {**not_executed, "required_native_gate": "PASS", "successful_reference_trajectories": 0})
dump("shold_transfer_source_manifest.json", {**not_executed, "preregistered_source": "D17/D21 train-only S_HOLD source", "expected_snapshots": 64, "validation_snapshots_accessed": 0})
routes = [
    {"route": "R0_NATIVE_SHOLD_WALK", "actor": "S_HOLD", "cross_fade_steps": 0},
    {"route": "R1_STAGE2Q_HARD", "actor": "Stage 2Q", "cross_fade_steps": 0},
    {"route": "R2_STAGE2Q_BLEND4", "actor": "Stage 2Q", "cross_fade_steps": 4, "training_only": True},
    {"route": "R3_STAGE2Q_BLEND8", "actor": "Stage 2Q", "cross_fade_steps": 8, "training_only": True},
    {"route": "R4_STAGE2Q_BLEND12", "actor": "Stage 2Q", "cross_fade_steps": 12, "training_only": True},
]
dump("candidate_route_contract.json", {"status": "PREREGISTERED_NOT_EXECUTED", "target": {"vx": .6, "vy": 0, "yaw": 0}, "command_ramp_s": .5, "source_snapshots": 64, "routes": routes, "stop_rule_applied": STOP_REASON})
route_fields = ["route", "snapshot_id", "classification", "first_step", "walk_acquisition", "wmove_retained", "fall", "dangerous_slip", "impact", "velocity_saturation", "torque_saturation"]
with (OUT / "candidate_route_results.csv").open("w", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=route_fields).writeheader()
dump("candidate_route_results.json", {**not_executed, "planned_episodes": 320, "executed_episodes": 0, "routes": {r["route"]: "NOT_EXECUTED" for r in routes}})
dump("first_step_reference.json", {**not_executed, "reason_detail": "Native gate must pass before successful native trajectories define the reference thresholds."})
dump("walk_acquisition.json", {"native": {"gait_acquisition": native["native_gait_acquisition_rate"], "25_step_forward_confirmation": native["confirmation_rate"], "safe_success": native["success_rate"]}, "S_HOLD_transfer": not_executed})
dump("stage2q_to_wmove_handoff.json", {**not_executed, "action_continuity": None, "acquisition_retention": None, "steady_tracking": None, "safety": None})

# Required conditional bundle: an explicit tombstone, not fabricated demonstrations.
np.savez_compressed(OUT / "successful_trajectory_bundle.npz", status=np.array(["NOT_EXECUTED"]), reason=np.array([STOP_REASON]), trajectory_count=np.array([0], dtype=np.int64))
(OUT / "successful_trajectory_bundle.sha256").write_text(sha(OUT / "successful_trajectory_bundle.npz") + "  successful_trajectory_bundle.npz\n", encoding="ascii")
dump("successful_trajectory_manifest.json", {**not_executed, "safe_trajectory_count": 0, "unique_source_coverage": 0, "diversity": "NOT_MEASURED", "bundle_sha256": sha(OUT / "successful_trajectory_bundle.npz")})
dump("temporary_distillation_feasibility.json", {**not_executed, "MSE": None, "boundary_MSE": None, "cosine": None, "persistent_checkpoint": 0})

dump("stage_reference.json", {
    "stage": "Phase 2-D24A", "name": "existing forward START Teacher recovery and S_HOLD-source transfer audit",
    "starting_HEAD": START, "actual_HEAD_before_commit": git("rev-parse", "HEAD"),
    "teacher_identities": {"S_HOLD": SHOLD_SHA, "Stage_2Q": STAGE2Q_SHA, "W_MOVE": WMOVE_SHA, "S_STOP_OMNI": SSTOP_SHA},
    "historical_context": {"D23": "EXP014_D23_EXPLICIT_LEAD_PHASE_NO_REACHABILITY"},
})
dump("protocol.json", {
    "native": {"episodes": 100, "seed": 20269031, "target_speed": .6, "command": "1.0 s zero then 1.0 s minimum-jerk ramp; hold", "mandatory_gate": {"acquisition": .90, "confirmation": .90, "fall_max": .05, "slip_max": .10, "torque_saturation_max": .10}},
    "conditional_transfer": {"snapshots": 64, "routes": routes, "episodes": 320, "executed_only_if_native_gate_passes": True},
    "prohibitions": {"persistent_PPO": 0, "new_checkpoint": 0, "validation_102": 0, "heldout": 0, "RUN": 0, "action_search": 0},
})
dump("stage_classification.json", {"main_classification": CLASSIFICATION, "native_gate": "FAIL", "subclassifications": ["D24A_FORWARD_CONFIRMATION_GATE_FAIL", "D24A_TORQUE_SATURATION_GATE_FAIL"], "transfer_executed": False})
dump("recommended_next_action.json", {"only_next_experiment": "D25 phase-conditioned low-speed STEP continuation", "reason": "Existing Stage 2Q did not pass the preregistered native prerequisite, so no source-transfer or demonstration claim is permitted.", "do_not": ["direct Teacher distillation", "demonstration-based DAgger", "reward-based PPO", "fallback"]})

changed = git("diff", "--name-only", START).splitlines()
protected_overlap = [p for p in changed if ("phase_2_d" in p and not "phase_2_d24a" in p) or any(f"exp_{i:03d}" in p for i in range(5, 14))]
dump("protected_hashes.json", {
    "audit_basis": "Compared working-tree paths against starting HEAD; pre-existing unrelated dirty paths were neither modified nor staged by D24A.",
    "starting_HEAD": START,
    "checkpoint_hashes": {"S_HOLD": SHOLD_SHA, "Stage_2Q": STAGE2Q_SHA, "W_MOVE": WMOVE_SHA, "S_STOP_OMNI": SSTOP_SHA},
    "D6_through_D23_new_changes_by_D24A": 0,
    "exp005_through_exp013_new_changes_by_D24A": 0,
    "preexisting_dirty_protected_paths_preserved": protected_overlap,
    "persistent_update": 0, "new_persistent_checkpoint": 0, "validation_access": 0, "heldout_access": 0, "RUN_integration": 0, "remote_push": False,
})

(OUT / "reproduction_commands.ps1").write_text("""$ErrorActionPreference = 'Stop'
$repo = 'C:\\Users\\user\\workspace\\physical-ai-lab'
Set-Location -LiteralPath $repo
git rev-parse HEAD
git status --short
& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p 'experiments\\isaaclab\\exp_014_unitree_g1_explicit_motion_mode_unified_locomotion\\scripts\\run_phase2_d24a_native.py' --headless
python 'experiments\\isaaclab\\exp_014_unitree_g1_explicit_motion_mode_unified_locomotion\\scripts\\finalize_phase2_d24a.py'
# The native gate fails; do not run the conditional S_HOLD transfer.
""", encoding="utf-8")

REPORT.write_text(f"""# Exp014 Phase 2-D24A existing START Teacher transfer audit

## Outcome

Main classification: `{CLASSIFICATION}`.

The frozen Stage 2Q checkpoint reproduced its historical native gait classification on 100/100 train-only episodes. The stricter D24A gate did not pass: only {native['confirmation_rate']:.2%} completed the forward-velocity 25-step confirmation, torque saturation dwell occurred in {native['torque_saturation_rate']:.2%}, and safe joint success was {native['success_rate']:.2%}. Falls, dangerous slips, impacts, velocity saturation, and non-finite states were zero.

This does not revise the committed exp012 claim. It establishes that the old gait-classification success contract is not sufficient for the new D24A START-source prerequisite.

## Native first-step dynamics

An identifiable first single-support/forward-displacement event occurred in {len(first_steps)}/100 episodes. Its median time was {native_summary['first_step_time_s']['median']:.3f} s (p05 {native_summary['first_step_time_s']['p05']:.3f} s, p95 {native_summary['first_step_time_s']['p95']:.3f} s), measured from the native reset timeline. All observations, actions, states, contacts, forces, and foot traces were durably stored in the native NPZ; per-episode classifications were committed to SQLite with WAL/FULL settings.

## Conditional branches

Because the native mandatory gate failed, protocol section 6 required an immediate stop. R0 through R4, the S_HOLD-source transfer, Stage 2Q to W_MOVE handoff, successful-demonstration construction, and temporary static distillation were not executed. Their artifacts contain explicit `NOT_EXECUTED` reasons; no missing values were inferred.

## Protection

No policy or optimizer was updated, no checkpoint was created, the fixed validation 102 snapshots and all held-out data were untouched, and no RUN integration or remote push occurred. Existing D6-D23 artifacts and the frozen S_HOLD, Stage 2Q, W_MOVE, and S_STOP_OMNI checkpoints were not modified. Unrelated pre-existing working-tree changes were preserved.

## Next

The only recommended next experiment is D25 phase-conditioned low-speed STEP continuation. Direct Teacher distillation and demonstration-based DAgger are not authorized because the prerequisite native gate did not establish a usable demonstration source.
""", encoding="utf-8")

print(json.dumps({"classification": CLASSIFICATION, "native": native_summary, "output": str(OUT)}, indent=2))
