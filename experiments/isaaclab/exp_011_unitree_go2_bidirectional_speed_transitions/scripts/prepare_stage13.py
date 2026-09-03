"""Freeze Stage 13 protocol and capture repository provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage13_fresh_process_counterfactual_replay"
RAW = OUT / "raw"
START = "acf87bd2e30186df698f9aa77aafd98adb294ba0"


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
if head != START:
    raise RuntimeError(f"starting HEAD mismatch: expected {START}, got {head}")

stage12 = json.loads((
    REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
    "stage12_tangential_slip_reward_directionality/stage12_classification.json"
).read_text(encoding="utf-8"))
protocol = {
    "name": "EXP011_STAGE13_FRESH_PROCESS_COUNTERFACTUAL_V1",
    "frozen_before_execution": True,
    "process_contract": {
        "os_processes_per_run": 1,
        "isaac_app_lifecycles_per_process": 1,
        "episodes_per_process": 1,
        "action_variants_per_process": 1,
        "concurrency": 1,
        "same_lifecycle_reset_formal": False,
        "state_injection": False,
    },
    "seed_root": 20273901,
    "speeds_m_s": [0.2, 0.4, 0.6, 1.2, 2.0],
    "baseline_preflight": {"seeds_per_speed": 5, "repeats": 3, "runs": 75, "duration_s": 8.0},
    "branches": {"preflight_per_speed": 4, "formal_per_speed": 20, "formal_total": 100},
    "primary_delta": 0.02,
    "linearity_deltas": [0.01, 0.04],
    "linearity_fraction": 0.20,
    "action_dimensions": 12,
    "horizons_steps": [1, 2, 4, 8],
    "primary_horizon_steps": 8,
    "control_dt_s": 0.02,
    "locally_improving": {
        "slip_reduction_fraction": 0.20,
        "maximum_speed_error_degradation_m_s": 0.03,
        "maximum_heading_error_degradation_rad": 0.02,
        "contact_loss": False,
        "fall": False,
        "saturation_increase": False,
    },
    "stage12_gradient": {
        "q_g": 0.001453556353226304,
        "base_slip_cosine": -0.3287697434425354,
        "minibatch_pairwise_cosine_median": 0.3976132273674011,
    },
    "production_ppo_update": 0,
    "reward_optimization": 0,
}
protocol["sha256"] = canonical_hash(protocol)

(OUT / "starting_repository_state.json").write_text(
    json.dumps({
        "starting_head": head,
        "starting_status": status,
        "unrelated_dirty_paths": status,
    }, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
(OUT / "stage12_reference.json").write_text(
    json.dumps({
        "classification": stage12["classification"],
        "failure": stage12["primary_evidence"],
        "stage12_commit": START,
        "secondary_findings": stage12["secondary_findings_not_promoted_to_causal_classification"],
        "stage7_checkpoint_sha256": "d6bb5b7be94f0a827576256b6ae420cdc5b2267c389c7ba92951801f9e2899bd",
        "stage11_lambda_slip": 0.00559195994498,
        "stage10_heading_controller": {"kp": 1.0, "yaw_limit_rad_s": 0.10},
    }, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
(OUT / "protocol.json").write_text(
    json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps({"head": head, "protocol_sha256": protocol["sha256"]}, indent=2))
