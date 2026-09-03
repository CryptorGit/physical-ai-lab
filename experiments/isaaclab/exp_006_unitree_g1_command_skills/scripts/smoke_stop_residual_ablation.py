"""Pure-torch checks for STOP action-space ablation projections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from g1_command_skills.models import G1CommandResidualActor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    obs = TensorDict({"policy": torch.randn(2, 152)}, batch_size=[2])
    obs["policy"][:, 123:] = 0.0
    obs["policy"][:, 124] = 1.0
    obs["policy"][:, 130] = 1.0
    obs["policy"][:, 148] = 1.0
    actor = G1CommandResidualActor(
        obs, {"actor": ["policy"]}, "actor", 37,
        hidden_dims=[256, 128, 128], activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
        trainable_skill_ids=[],
    )
    checkpoint = torch.load(args.checkpoint.resolve(strict=True), map_location="cpu", weights_only=False)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    names = (
        "torso", "left_hip_yaw", "right_hip_yaw", "left_ankle_roll", "right_ankle_roll",
        "left_hip_roll", "right_hip_roll", "left_hip_pitch", "right_hip_pitch",
        "left_knee", "right_knee", "left_ankle_pitch", "right_ankle_pitch",
    )
    indices = {name: index for index, name in enumerate(names)}
    results = {}
    for mode in ("current", "yaw_mask", "yaw_ankle_roll_mask", "lateral_mask", "symmetric"):
        actor.configure_stop_residual_ablation(mode, indices)
        diagnostics = actor.diagnostic_components(obs)
        raw = diagnostics["selected_raw_residual"]
        masked = diagnostics["selected_residual"]
        zero_names = {
            "current": (),
            "yaw_mask": names[:3],
            "yaw_ankle_roll_mask": names[:5],
            "lateral_mask": names[:7],
            "symmetric": names[:7],
        }[mode]
        zeros_ok = all(torch.count_nonzero(masked[..., indices[name]]).item() == 0 for name in zero_names)
        symmetric_ok = True
        if mode == "symmetric":
            symmetric_ok = all(torch.equal(masked[..., indices[left]], masked[..., indices[right]]) for left, right in (
                ("left_hip_pitch", "right_hip_pitch"),
                ("left_knee", "right_knee"),
                ("left_ankle_pitch", "right_ankle_pitch"),
            ))
        results[mode] = {
            "raw_norm": float(raw.norm(dim=-1).mean()),
            "masked_norm": float(masked.norm(dim=-1).mean()),
            "required_zeros": zeros_ok,
            "required_symmetry": symmetric_ok,
        }
    result = {"conditions": results, "passed": all(x["required_zeros"] and x["required_symmetry"] for x in results.values())}
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("STOP residual ablation smoke failed")


if __name__ == "__main__":
    main()
