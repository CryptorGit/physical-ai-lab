from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np

from env import OpenDuckCalibratedWalkEnv


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Candidate:
    period: float
    hip_sine: float
    hip_second: float
    knee_swing: float
    knee_bias: float
    ankle_sine: float
    roll_sine: float
    roll_phase: float


def candidate_action(candidate: Candidate, step: int) -> np.ndarray:
    phase = 2.0 * np.pi * step / candidate.period
    sine = np.sin(phase)
    hip = (
        candidate.hip_sine * sine
        + candidate.hip_second * np.sin(2.0 * phase)
    )
    roll = candidate.roll_sine * np.sin(phase + candidate.roll_phase)
    left_swing = 0.5 + 0.5 * sine
    right_swing = 0.5 - 0.5 * sine

    action = np.zeros(10, dtype=np.float32)
    action[1] = roll
    action[2] = hip
    action[3] = candidate.knee_bias + candidate.knee_swing * left_swing
    action[4] = candidate.ankle_sine * sine
    action[6] = roll
    action[7] = -hip
    action[8] = candidate.knee_bias + candidate.knee_swing * right_swing
    action[9] = -candidate.ankle_sine * sine
    return np.clip(action, -1.0, 1.0)


def evaluate(candidate: Candidate, *, steps: int = 400) -> dict:
    warmup = 50
    env = OpenDuckCalibratedWalkEnv(
        seed=29,
        episode_steps=steps,
        command_velocity=0.10,
    )
    _, _ = env.reset()
    start_x = float(env.data.qpos[0])
    min_upright = 1.0
    terminated = False
    completed = 0

    for step in range(steps):
        action = (
            np.zeros(10, dtype=np.float32)
            if step < warmup
            else candidate_action(candidate, step - warmup)
        )
        _, _, terminated, truncated, _ = env.step(action)
        completed = step + 1
        min_upright = min(min_upright, float(env._base_rotation()[2, 2]))
        if terminated or truncated:
            break

    distance_x = float(env.data.qpos[0] - start_x)
    lateral = abs(float(env.data.qpos[1]))
    completion = completed / steps
    objective = (
        25.0 * distance_x
        + 5.0 * completion
        + 0.5 * max(0.0, min_upright)
        - 5.0 * lateral
        - (20.0 if terminated else 0.0)
    )
    env.close()
    return {
        "candidate": asdict(candidate),
        "objective": objective,
        "distance_x_m": distance_x,
        "lateral_m": lateral,
        "steps": completed,
        "terminated": terminated,
        "min_upright_cosine": min_upright,
    }


def sample_candidates(count: int, seed: int) -> list[Candidate]:
    rng = np.random.default_rng(seed)
    return [
        Candidate(
            period=float(rng.uniform(30.0, 80.0)),
            hip_sine=float(rng.uniform(-1.0, 1.0)),
            hip_second=float(rng.uniform(-0.5, 0.5)),
            knee_swing=float(rng.uniform(-1.0, 1.0)),
            knee_bias=float(rng.uniform(-0.3, 0.3)),
            ankle_sine=float(rng.uniform(-1.0, 1.0)),
            roll_sine=float(rng.uniform(-0.6, 0.6)),
            roll_phase=float(rng.uniform(-np.pi, np.pi)),
        )
        for _ in range(count)
    ]


def mutate_candidates(
    elites: list[Candidate],
    count: int,
    rng: np.random.Generator,
    scale: float,
) -> list[Candidate]:
    result = []
    for _ in range(count):
        parent = elites[int(rng.integers(0, len(elites)))]
        result.append(
            Candidate(
                period=float(
                    np.clip(parent.period + rng.normal(0, 8 * scale), 25, 90)
                ),
                hip_sine=float(
                    np.clip(parent.hip_sine + rng.normal(0, 0.20 * scale), -1, 1)
                ),
                hip_second=float(
                    np.clip(
                        parent.hip_second + rng.normal(0, 0.15 * scale),
                        -0.7,
                        0.7,
                    )
                ),
                knee_swing=float(
                    np.clip(
                        parent.knee_swing + rng.normal(0, 0.20 * scale), -1, 1
                    )
                ),
                knee_bias=float(
                    np.clip(
                        parent.knee_bias + rng.normal(0, 0.08 * scale),
                        -0.4,
                        0.4,
                    )
                ),
                ankle_sine=float(
                    np.clip(
                        parent.ankle_sine + rng.normal(0, 0.20 * scale), -1, 1
                    )
                ),
                roll_sine=float(
                    np.clip(
                        parent.roll_sine + rng.normal(0, 0.15 * scale),
                        -0.8,
                        0.8,
                    )
                ),
                roll_phase=float(
                    (parent.roll_phase + rng.normal(0, 0.5 * scale) + np.pi)
                    % (2 * np.pi)
                    - np.pi
                ),
            )
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=480)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--refinements", type=int, default=4)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = HERE / "artifacts" / "reference_search"
    output_path = output_dir / "results.json"
    if args.resume and output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        previous_elites = [
            Candidate(**item["candidate"]) for item in previous["top"][:20]
        ]
        candidates = mutate_candidates(
            previous_elites,
            args.candidates,
            rng,
            scale=0.5,
        )
    else:
        candidates = sample_candidates(args.candidates, args.seed)
    all_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for iteration in range(args.refinements + 1):
            evaluator = partial(evaluate, steps=args.steps)
            results = list(executor.map(evaluator, candidates, chunksize=1))
            all_results.extend(results)
            all_results.sort(
                key=lambda item: item["objective"], reverse=True
            )
            best = all_results[0]
            print(
                f"iteration={iteration} objective={best['objective']:.4f} "
                f"distance={best['distance_x_m']:.4f} "
                f"steps={best['steps']}",
                flush=True,
            )
            if iteration < args.refinements:
                elites = [
                    Candidate(**item["candidate"])
                    for item in all_results[:32]
                ]
                candidates = mutate_candidates(
                    elites,
                    args.candidates,
                    rng,
                    scale=0.75**iteration,
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "candidate_count": args.candidates * (args.refinements + 1),
                "workers": args.workers,
                "top": all_results[:20],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(all_results[:5], indent=2))
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
