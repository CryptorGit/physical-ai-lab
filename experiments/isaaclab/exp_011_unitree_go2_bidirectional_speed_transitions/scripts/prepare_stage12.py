"""Freeze the exp_011 Stage 12 diagnostic protocol before any rollout."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage12_tangential_slip_reward_directionality"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "raw").mkdir(exist_ok=True)


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
expected = "fed3b08e187b29b5bcbf14e983dd29e60a35b4d4"
if head != expected:
    raise SystemExit(f"starting HEAD mismatch: expected {expected}, got {head}")

protocol = {
    "name": "EXP011_STAGE12_TANGENTIAL_SLIP_DIRECTIONALITY_V1",
    "frozen_before_rollout": True,
    "diagnostic_seed_root": 20272901,
    "checkpoints": [0, 1, 10, 25, 50, 75, 100, 150, 200],
    "steady": {
        "speeds_m_s": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
        "episodes": 100,
        "duration_s": 8.0,
    },
    "transitions": {
        "pairs_m_s": [
            [0.0, 0.2], [0.0, 0.4], [0.0, 0.6],
            [0.6, 0.4], [0.6, 0.2], [0.6, 0.0],
            [0.0, 1.2], [1.2, 2.0], [2.0, 1.2], [1.2, 0.0],
        ],
        "episodes": 50,
        "source_hold_s": 3.0,
        "ramp_s": 1.5,
        "target_hold_s": 5.0,
    },
    "discount": "official Stage 7 runner value resolved at runtime",
    "gae_lambda": "official Stage 7 runner value resolved at runtime",
    "diagnostic_value_model": {
        "architecture": [48, 128, 128, 1],
        "activation": "ELU",
        "episode_split": [0.70, 0.15, 0.15],
    },
    "minibatch_permutations": 100,
    "counterfactual": {
        "speeds_m_s": [0.2, 0.4, 0.6, 1.2, 2.0],
        "states_per_speed": 100,
        "normalized_action_delta": 0.02,
        "linearity_fraction": 0.20,
        "linearity_deltas": [0.01, 0.04],
        "horizons_steps": [1, 2, 4, 8],
        "primary_horizon_steps": 8,
        "state_setter": False,
    },
    "production_ppo_update": 0,
    "reward_optimization": 0,
}
canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
protocol["sha256"] = hashlib.sha256(canonical).hexdigest()
dump("protocol.json", protocol)
dump("stage11_reference.json", {
    "classification": "GO2_TANGENTIAL_SLIP_NO_EFFECT",
    "stage7_parent_sha256": "d6bb5b7be94f0a827576256b6ae420cdc5b2267c389c7ba92951801f9e2899bd",
    "stage11_selected_iteration": 0,
    "stage11_selected_sha256": "e7c6eb71b943369360686deeb376881161c6f78ce108ee29d89040a6a6ae464f",
    "lambda_slip": 0.005591959944980788,
    "slip_protocol_sha256": "74b46a8ed230d4531259ff1ec52ef9937d308ec3a1334b9feeaa5a10707d0f83",
    "stage10_controller_sha256": "47a2dc2608fabf6e1ab5efad3776634b538ae2a895ea93658751ccb049d558f1",
})
dump("diagnostic_seed_manifest.json", {
    "root": 20272901,
    "steady_episode_seeds": list(range(20272901, 20273001)),
    "transition_episode_seeds": list(range(20272901, 20272951)),
    "counterfactual_episode_seeds": list(range(20272901, 20273001)),
    "selection_from_success": False,
})
dump("starting_repository_state.json", {
    "starting_head": head,
    "starting_status": status,
    "unrelated_dirty_paths": status,
})
print(protocol["sha256"])
