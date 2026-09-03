"""Frozen-checkpoint evaluation suites for the Phase W1A3 diagnosis."""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a3_rear_left_low_speed_retention_diagnosis"
W1A = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a_all_direction_translation_walk/checkpoints/model_120.pt"
W1A2 = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_160.pt"

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("timeline", "boundary", "interpolation", "tradeoff"), required=True)
parser.add_argument("--checkpoint")
parser.add_argument("--tag", required=True)
parser.add_argument("--lambda-value", type=float)
args, launcher = parser.parse_known_args()


def interpolated_checkpoint(value: float) -> Path:
    left = torch.load(W1A, map_location="cpu", weights_only=False)
    right = torch.load(W1A2, map_location="cpu", weights_only=False)
    state = {}
    for key, tensor in left["actor_state_dict"].items():
        if key.startswith("distribution."):
            state[key] = tensor.clone()
        else:
            state[key] = tensor.mul(1.0 - value).add(right["actor_state_dict"][key], alpha=value)
    handle = tempfile.NamedTemporaryFile(prefix="exp013_w1a3_interp_", suffix=".pt", delete=False)
    handle.close()
    path = Path(handle.name)
    torch.save({"actor_state_dict": state}, path)
    return path


temporary = None
checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
if args.mode == "interpolation":
    if args.lambda_value is None:
        raise ValueError("--lambda-value is required")
    temporary = interpolated_checkpoint(args.lambda_value)
    checkpoint = temporary
if checkpoint is None:
    raise ValueError("--checkpoint is required")

sys.argv = [
    "evaluate_w1a.py", "--suite", "formal", "--checkpoint", str(checkpoint),
    "--tag", args.tag, *launcher,
]
sys.path.insert(0, str(HERE.parent))
import evaluate_w1a as base  # noqa: E402

base.OUT = OUT


def conditions():
    if args.mode == "timeline":
        pairs = [
            (.3, 225), (.3, 247.5), (.6, 225), (.6, 247.5),
            (.3, 135), (.3, 112.5), (.6, 135), (.6, 112.5),
            (.3, 180), (.3, 270), (.6, 0), (1.2, 0),
        ]
        return [base.static(f"S{s:.2f}_D{d:06.2f}", s, d, 50) for s, d in pairs]
    if args.mode == "boundary":
        return [
            base.static(f"S{s:.2f}_D{d:06.2f}", s, d, 20)
            for d in (180, 191.25, 202.5, 213.75, 225, 236.25, 247.5, 258.75, 270)
            for s in (.20, .25, .30, .35, .40, .45, .50, .55, .60)
        ]
    if args.mode == "tradeoff":
        return [
            base.static(f"S{s:.2f}_D{d:06.2f}", s, d, 50)
            for s in (.3, .6) for d in (index * 22.5 for index in range(16))
        ] + [base.static("S1.20_D000.00", 1.2, 0, 50)]
    return [
        base.static(f"S{s:.2f}_D{d:06.2f}", s, d, 20)
        for s in (.3, .6) for d in (index * 22.5 for index in range(16))
    ] + [base.static("S1.20_D000.00", 1.2, 0, 20)]


base.conditions = conditions
try:
    base.main()
finally:
    if temporary is not None:
        temporary.unlink(missing_ok=True)
