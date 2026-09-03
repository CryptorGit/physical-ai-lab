"""Freeze exp_011 Stage 10 protocol and run offline controller tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage10_phase_gated_fixed_heading"
sys.path.insert(0, str(EXP / "src"))

from go2_bidirectional.phase_gated_heading import run_unit_tests  # noqa: E402

START = "b573b730fb5b0e5447cbdce250d5cb49c95ae6f7"
SEED = 20269901
CHECKPOINTS = {
    "official_parent": (
        REPO / ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/"
        "Assets/Isaac/6.0/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/"
        "Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt"
    ),
    "stage4_selected": (
        REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
        "stage4_resumed_optimizer_training/checkpoints/model_50.pt"
    ),
    "stage7_selected": (
        REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
        "stage7_low_speed_gait_stabilization/checkpoints/model_50.pt"
    ),
    "stage6_protocol": (
        REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
        "stage6_corrected_endpoint_formal/protocol_hash.json"
    ),
}
EXPECTED = {
    "official_parent": "32039715f0892650691aa8d5c50233e7c4b858469d87114ef92794e2c65b59c0",
    "stage4_selected": "e2a3de144984683efcc7b4fe451898c3d2b450a7ae3696ad6784a027a9756bea",
    "stage7_selected": "d6bb5b7be94f0a827576256b6ae420cdc5b2267c389c7ba92951801f9e2899bd",
    "stage6_protocol": "d10f0bf7809046afa1b72e663a861acc53e45f9460aec18365c687de140c0908",
}


def dump(name: str, value) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
stage10_markers = (
    "stage10_phase_gated_fixed_heading", "prepare_stage10.py",
    "evaluate_stage10_heading_controller.py", "finalize_stage10.py",
    "run_stage10_heading_diagnosis.ps1", "phase_gated_heading.py",
    "exp_011_go2_phase_gated_fixed_heading_report.md",
    "play_exp011_go2_bidirectional.py", "play_exp011_go2_bidirectional.ps1",
    "stage10_run_", "stage10_c1_", "stage10_c2_",
)
status = [line for line in status if not any(marker in line for marker in stage10_markers)]
if head != START:
    raise SystemExit(f"unexpected starting HEAD: {head}")

dump("starting_repository_state.json", {
    "starting_head": head,
    "starting_status": status,
    "unrelated_dirty_paths": status,
})
dump("stage9_reference.json", {
    "classification": "GO2_CONTACT_KINEMATICS_NOT_PRIMARY",
    "heading_interpretation": "ABSOLUTE_HEADING_UNOBSERVABILITY_REMAINS",
    "next_action": "phase-gated fixed-heading command controller diagnosis",
    "stage9_commit": START,
})
protocol = {
    "stage": 10,
    "target": "PHASE_GATED_FIXED_HEADING",
    "checkpoint": "stage7_selected",
    "checkpoint_sha256": EXPECTED["stage7_selected"],
    "controllers": ["OPEN_LOOP", "ALWAYS_ON_FIXED_HEADING", "PHASE_GATED_FIXED_HEADING"],
    "feedback": {"kp": 1.0, "omega_max_rad_s": 0.10, "grid_search": False},
    "steady_gate": {
        "disabled_s": [0.0, 1.0],
        "reference_window_s": [0.5, 1.0],
        "activation_s": [1.0, 1.5],
        "activation_profile": "minimum_jerk",
    },
    "transition_gate": {
        "source_hold_s": 3.0,
        "speed_ramp_s": 1.5,
        "target_hold_s": 5.0,
        "reference_window_s": [2.5, 3.0],
        "acquisition_hold_s": 0.5,
        "activation_s": 0.5,
        "latch_after_active": True,
    },
    "seed_root": SEED,
    "episodes": 50,
    "deterministic_policy": True,
    "ppo_updates": 0,
    "reward_optimization": 0,
    "policy_gradient": 0,
    "production_status": "DIAGNOSTIC_CANDIDATE",
}
dump("protocol.json", protocol)
dump("fixed_heading_contract.json", {
    "error": "atan2(sin(reference-current), cos(reference-current))",
    "quaternion": "xyzw via Stage 6 shared helper",
    "kp": 1.0,
    "omega_max_rad_s": 0.10,
    "deadband": None,
    "integral": None,
    "derivative": None,
    "policy_parameters_modified": False,
})
dump("phase_gate_contract.json", {
    "states": [
        "DISABLED_SOURCE", "DISABLED_RAMP", "WAIT_TARGET_ACQUISITION",
        "ACTIVATING", "ACTIVE", "TERMINATED",
    ],
    "range": [0.0, 1.0],
    "steady": protocol["steady_gate"],
    "transition": protocol["transition_gate"],
    "acquisition_failure": "gate remains zero",
    "active_latch": True,
})
tests = run_unit_tests()
dump("phase_gated_heading_unit_tests.json", tests)
if not tests["all_pass"]:
    raise SystemExit("phase-gated heading unit test failure")
dump("stage10_seed_manifest.json", {
    "seed_root": SEED,
    "selection": "pre-generated consecutive seeds, shared by C0/C1/C2",
    "episode_seeds": list(range(SEED, SEED + 50)),
    "success_selection": False,
})
(OUT / "reproduction_commands.ps1").write_text(
    """$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\\..\\..\\..\\..")).Path
$script = Join-Path $repo "experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\run_stage10_heading_diagnosis.ps1"
Push-Location $repo
try {
  & $script -Device "cuda:0"
} finally {
  Pop-Location
}
""",
    encoding="utf-8",
)
actual = {name: sha(path) for name, path in CHECKPOINTS.items() if name != "stage6_protocol"}
protocol_hash_record = json.loads(CHECKPOINTS["stage6_protocol"].read_text(encoding="utf-8"))
actual["stage6_protocol"] = protocol_hash_record["sha256"]
dump("preflight_protected_hashes.json", {
    "expected": EXPECTED, "actual": actual,
    "all_match": actual == EXPECTED,
})
if actual != EXPECTED:
    raise SystemExit("protected hash mismatch")
print("STAGE10_PROTOCOL_FROZEN")
