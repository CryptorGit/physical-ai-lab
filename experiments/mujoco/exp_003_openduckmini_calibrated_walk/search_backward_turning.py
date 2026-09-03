"""Grid-search asymmetric reverse gait amplitudes for turning."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from evaluate_official_policy import OfficialPolicyEvaluator


def parse_args() -> argparse.Namespace:
    experiment = Path(__file__).resolve().parent
    workspace = experiment.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=Path,
        default=workspace
        / ".openduck_playground_source_review"
        / "playground"
        / "open_duck_mini_v2"
        / "xmls"
        / "scene_flat_terrain_backlash_calibrated.xml",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=experiment / "artifacts" / "calibrated_hybrid_policy_v22.onnx",
    )
    parser.add_argument(
        "--reference-data",
        type=Path,
        default=workspace
        / ".openduck_playground_source_review"
        / "playground"
        / "open_duck_mini_v2"
        / "data"
        / "polynomial_coefficients_calibrated.pkl",
    )
    parser.add_argument("--yaw", type=float, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--grid-min", type=float, default=-3.0)
    parser.add_argument("--grid-max", type=float, default=3.0)
    parser.add_argument("--grid-count", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = OfficialPolicyEvaluator(
        args.scene, args.policy, args.reference_data
    )
    values = np.linspace(args.grid_min, args.grid_max, args.grid_count)
    command = (-0.07, 0.0, args.yaw)
    candidates = []
    for left, right in itertools.product(values, repeat=2):
        evaluator.backward_yaw_amplitude_gains = np.asarray([left, right])
        episodes = [
            evaluator.run_episode(
                command,
                args.seconds,
                seed=seed,
                initial_joint_noise=0.02,
                initial_base_speed=0.05,
                positive_yaw_lateral_compensation=0.06,
            )
            for seed in range(args.episodes)
        ]
        fall_rate = float(np.mean([episode.fell for episode in episodes]))
        velocity = np.mean(
            np.asarray([episode.mean_velocity_xyz for episode in episodes]),
            axis=0,
        )
        yaw_rate = float(np.mean([episode.mean_yaw_rate for episode in episodes]))
        minimum_upright = float(
            min(episode.minimum_upright for episode in episodes)
        )
        score = (
            100.0 * fall_rate
            + 3.0 * abs(yaw_rate - args.yaw)
            + 2.0 * abs(velocity[0] - command[0])
            + abs(velocity[1])
            + 10.0 * max(0.0, 0.9 - minimum_upright)
        )
        candidates.append(
            {
                "left_gain": float(left),
                "right_gain": float(right),
                "score": float(score),
                "fall_rate": fall_rate,
                "mean_velocity_xyz": velocity.tolist(),
                "mean_yaw_rate": yaw_rate,
                "minimum_upright": minimum_upright,
            }
        )
    candidates.sort(key=lambda candidate: candidate["score"])
    payload = {
        "command": list(command),
        "episodes_per_candidate": args.episodes,
        "seconds": args.seconds,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["candidates"][:10], indent=2))


if __name__ == "__main__":
    main()
