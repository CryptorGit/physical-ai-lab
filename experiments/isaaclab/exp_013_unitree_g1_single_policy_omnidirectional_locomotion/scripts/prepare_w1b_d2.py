"""Prepare the read-only EXP013 W1B-D2 diagnosis contract."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_d2_yaw_rate_tracking_boundary_diagnosis"
)
R2 = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun"
)
OUT.mkdir(parents=True, exist_ok=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name: str, payload) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True)
unrelated_status = [
    line for line in status.splitlines()
    if "w1b_d2" not in line
    and "phase_w1b_d2_yaw_rate_tracking_boundary_diagnosis" not in line
    and "exp_013_g1_phase_w1b_d2_yaw_rate_tracking_boundary_report.md" not in line
]
log = subprocess.check_output(
    ["git", "log", "--oneline", "--decorate", "-25"], cwd=REPO, text=True
)
manifest = json.loads((R2 / "checkpoint_manifest.json").read_text(encoding="utf-8"))
selected = json.loads((R2 / "selected_checkpoint.json").read_text(encoding="utf-8"))

dump("stage_reference.json", {
    "stage": "Phase W1B-D2 yaw-rate tracking boundary diagnosis",
    "starting_head_reported": "c71b39ee985f3a424b4b0a13e7242d2b49bd9c2a",
    "starting_head_actual": head,
    "starting_head_matches": head == "c71b39ee985f3a424b4b0a13e7242d2b49bd9c2a",
    "starting_status_short": unrelated_status,
    "starting_log_25": log.splitlines(),
    "selected_checkpoint": selected["path"],
    "selected_checkpoint_sha256": selected["sha256"],
    "read_only_policy_diagnosis": True,
    "new_persistent_policy_checkpoint": 0,
    "remote_push": False,
})
dump("protocol.json", {
    "stage": "W1B-D2",
    "objective": "diagnose positive-yaw rate tracking boundary",
    "checkpoint_updates": 0,
    "optimizer_steps": 0,
    "training": False,
    "formal_command_calibration_adopted": False,
    "mirror_wrapper_runtime_adopted": False,
    "seed": 20276021,
    "episode_duration_s": 8,
})
dump("checkpoint_manifest.json", {
    "source": str(R2 / "checkpoint_manifest.json"),
    "entries": manifest["entries"],
    "selected_iteration": 200,
    "selected_sha256": selected["sha256"],
    "all_checkpoints_read_only": True,
})
