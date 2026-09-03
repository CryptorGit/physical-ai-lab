#!/usr/bin/env python3
"""Analyze uninterrupted versus fresh-process resumed PPO trial distributions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import jax
import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/user/openduck_training_backward_v23_20260729")
sys.path[:0] = [str(EXPERIMENT), str(SOURCE)]

from training.checkpointing import load_checkpoint  # noqa: E402


PRIMARY = (
    "actor_delta_l2",
    "critic_delta_l2",
    "adam_first_moment_norm",
    "adam_second_moment_norm",
    "policy_loss",
    "value_loss",
    "approx_kl",
    "entropy",
    "mean_return",
    "fall_rate",
    "termination_rate",
    "tracking_rmse",
)


def _leaves(tree: Any) -> list[np.ndarray]:
    return [np.asarray(x, dtype=np.float64) for x in jax.tree_util.tree_leaves(tree)]


def l2(tree: Any) -> float:
    return float(np.sqrt(sum(np.sum(np.square(x)) for x in _leaves(tree))))


def delta_l2(left: Any, right: Any) -> float:
    return float(
        np.sqrt(
            sum(
                np.sum(np.square(a - b))
                for a, b in zip(_leaves(left), _leaves(right))
            )
        )
    )


def cosine(left: Any, right: Any) -> float:
    pairs = list(zip(_leaves(left), _leaves(right)))
    numerator = sum(np.sum(a * b) for a, b in pairs)
    left_norm = np.sqrt(sum(np.sum(a * a) for a, _ in pairs))
    right_norm = np.sqrt(sum(np.sum(b * b) for _, b in pairs))
    return float(numerator / max(left_norm * right_norm, 1e-30))


def read_telemetry(trial: dict[str, str]) -> list[dict[str, Any]]:
    result = []
    if trial["mode"] == "R":
        result.extend(
            json.loads(
                (
                    Path(trial["midpoint_checkpoint"])
                    / "update_telemetry.json"
                ).read_text()
            )
        )
    result.extend(
        json.loads(
            (
                Path(trial["final_checkpoint"]) / "update_telemetry.json"
            ).read_text()
        )
    )
    if len(result) != 4:
        raise ValueError(f'{trial["trial_id"]}: expected four telemetry updates')
    return result


def endpoint_row(initial_state: Any, trial: dict[str, str]) -> dict[str, Any]:
    final_state, _, _ = load_checkpoint(Path(trial["final_checkpoint"]))
    initial_params = initial_state.learner_state.params
    final_params = final_state.learner_state.params
    telemetry = read_telemetry(trial)
    optimizer = [update["optimizer"] for update in telemetry]
    rollout = [update["rollout"] for update in telemetry]
    adam_state = final_state.learner_state.optimizer_state[1][0]

    sample_total = sum(float(x["rollout_sample_total"]) for x in rollout)
    reward_sum = sum(float(x["reward_sum"]) for x in rollout)
    fall_count = sum(float(x["fall_count"]) for x in rollout)
    termination_count = sum(float(x["termination_count"]) for x in rollout)
    tracking_sse = sum(
        float(x["tracking_squared_error_sum"]) for x in rollout
    )
    histogram = np.sum(
        np.asarray([x["joint_command_head_histogram"] for x in rollout]),
        axis=0,
    )
    nonfinite = sum(
        float(x["gradient_nonfinite_count"])
        + float(x["parameter_nonfinite_count"])
        for x in optimizer
    )
    # A fixed two-step diagnostic horizon is used here.  This is not an episode
    # return and is explicitly named in the contract/report.
    mean_two_step_return = reward_sum / max(sample_total, 1.0) * 2.0
    return {
        "trial_id": trial["trial_id"],
        "mode": trial["mode"],
        "status": trial["status"],
        "actor_delta_l2": delta_l2(
            initial_params.policy, final_params.policy
        ),
        "critic_delta_l2": delta_l2(
            initial_params.value, final_params.value
        ),
        "actor_cosine_to_initial": cosine(
            initial_params.policy, final_params.policy
        ),
        "critic_cosine_to_initial": cosine(
            initial_params.value, final_params.value
        ),
        "actor_parameter_norm": l2(final_params.policy),
        "critic_parameter_norm": l2(final_params.value),
        "adam_first_moment_norm": l2(adam_state.mu),
        "adam_second_moment_norm": l2(adam_state.nu),
        "effective_adam_step_norm": float(
            np.mean([x["effective_adam_step_scale"] for x in optimizer])
        ),
        "global_update_norm": delta_l2(initial_params, final_params),
        "policy_loss": float(np.mean([x["policy_loss"] for x in optimizer])),
        "value_loss": float(np.mean([x["v_loss"] for x in optimizer])),
        "entropy": float(np.mean([x["entropy_loss"] for x in optimizer])),
        "approx_kl": float(np.mean([x["kl_mean"] for x in optimizer])),
        "explained_variance": float(
            np.mean([x["explained_variance"] for x in telemetry])
        ),
        "gradient_norm": float(
            np.mean([x["global_gradient_norm"] for x in optimizer])
        ),
        "mean_return": mean_two_step_return,
        "fall_rate": fall_count / max(sample_total, 1.0),
        "termination_rate": termination_count / max(sample_total, 1.0),
        "tracking_rmse": float(
            np.sqrt(tracking_sse / max(sample_total * 3.0, 1.0))
        ),
        "nonfinite_count": nonfinite,
        "joint_command_histogram": json.dumps(histogram.astype(int).tolist()),
        "formal_command_count": int(
            sum(
                np.sum(np.asarray(x["command_step_count"])[:19])
                for x in rollout
            )
        ),
        "off_grid_count": int(
            sum(np.asarray(x["command_step_count"])[19] for x in rollout)
        ),
    }


def iqm(values: np.ndarray) -> float:
    ordered = np.sort(values)
    trim = int(np.floor(0.25 * ordered.size))
    kept = ordered[trim : ordered.size - trim] if trim else ordered
    return float(np.mean(kept))


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, draws: int = 10000
) -> tuple[float, float]:
    sampled = rng.choice(values, size=(draws, values.size), replace=True)
    means = np.mean(sampled, axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def bootstrap_difference_ci(
    uninterrupted: np.ndarray,
    resumed: np.ndarray,
    rng: np.random.Generator,
    draws: int = 10000,
) -> tuple[float, float]:
    left = rng.choice(
        uninterrupted, size=(draws, uninterrupted.size), replace=True
    )
    right = rng.choice(resumed, size=(draws, resumed.size), replace=True)
    differences = np.mean(right, axis=1) - np.mean(left, axis=1)
    return tuple(float(x) for x in np.quantile(differences, [0.025, 0.975]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--trial-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    trials = list(csv.DictReader(args.trial_manifest.open()))
    completed = [x for x in trials if x["status"] == "COMPLETED"]
    initial_state, _, _ = load_checkpoint(args.initial)
    rows = [endpoint_row(initial_state, trial) for trial in completed]
    write_csv(args.output / "resume_primary_metrics.csv", rows)

    rng = np.random.default_rng(20260730)
    comparisons = []
    effects = {}
    for metric in PRIMARY:
        u = np.asarray(
            [float(x[metric]) for x in rows if x["mode"] == "U"]
        )
        r = np.asarray(
            [float(x[metric]) for x in rows if x["mode"] == "R"]
        )
        if not u.size or not r.size:
            continue
        pooled = np.sqrt(
            ((u.size - 1) * np.var(u, ddof=1) + (r.size - 1) * np.var(r, ddof=1))
            / max(u.size + r.size - 2, 1)
        )
        difference = float(np.mean(r) - np.mean(u))
        effect = (
            difference / pooled
            if pooled > 0
            else (0.0 if difference == 0 else float("inf"))
        )
        effects[metric] = effect
        u_ci = bootstrap_mean_ci(u, rng)
        r_ci = bootstrap_mean_ci(r, rng)
        diff_ci = bootstrap_difference_ci(u, r, rng)
        comparisons.append(
            {
                "metric": metric,
                "u_mean": np.mean(u),
                "u_median": np.median(u),
                "u_iqm": iqm(u),
                "u_std": np.std(u, ddof=1),
                "u_ci95_low": u_ci[0],
                "u_ci95_high": u_ci[1],
                "r_mean": np.mean(r),
                "r_median": np.median(r),
                "r_iqm": iqm(r),
                "r_std": np.std(r, ddof=1),
                "r_ci95_low": r_ci[0],
                "r_ci95_high": r_ci[1],
                "mean_difference_r_minus_u": difference,
                "difference_ci95_low": diff_ci[0],
                "difference_ci95_high": diff_ci[1],
                "standardized_mode_effect": effect,
                "probability_r_less_than_u": float(
                    np.mean(r[:, None] < u[None, :])
                    + 0.5 * np.mean(r[:, None] == u[None, :])
                ),
            }
        )
    write_csv(args.output / "resume_distribution_comparison.csv", comparisons)

    u_rows = [x for x in rows if x["mode"] == "U"]
    r_rows = [x for x in rows if x["mode"] == "R"]
    u_hist = np.sum(
        [np.asarray(json.loads(x["joint_command_histogram"])) for x in u_rows],
        axis=0,
    )
    r_hist = np.sum(
        [np.asarray(json.loads(x["joint_command_histogram"])) for x in r_rows],
        axis=0,
    )
    u_dist = u_hist / max(np.sum(u_hist), 1)
    r_dist = r_hist / max(np.sum(r_hist), 1)
    tv = float(0.5 * np.sum(np.abs(u_dist - r_dist)))

    crash_u = sum(x["status"] != "COMPLETED" for x in trials if x["mode"] == "U")
    crash_r = sum(x["status"] != "COMPLETED" for x in trials if x["mode"] == "R")
    nonfinite = sum(float(x["nonfinite_count"]) for x in rows)
    ratios = {
        metric: float(
            np.median([float(x[metric]) for x in r_rows])
            / max(np.median([float(x[metric]) for x in u_rows]), 1e-30)
        )
        for metric in ("actor_delta_l2", "critic_delta_l2")
    }
    comparison_by_metric = {x["metric"]: x for x in comparisons}
    fall_upper = comparison_by_metric["fall_rate"]["difference_ci95_high"]
    termination_upper = comparison_by_metric["termination_rate"][
        "difference_ci95_high"
    ]
    pass_checks = {
        "twenty_completed_each": len(u_rows) == 20 and len(r_rows) == 20,
        "nonfinite_zero": nonfinite == 0,
        "crash_rate_equal": crash_u == crash_r,
        "all_standardized_effects_le_0p25": all(
            abs(value) <= 0.25 for value in effects.values()
        ),
        "actor_delta_median_ratio": 0.95
        <= ratios["actor_delta_l2"]
        <= 1.05,
        "critic_delta_median_ratio": 0.95
        <= ratios["critic_delta_l2"]
        <= 1.05,
        "fall_difference_ci_upper_le_0p01": fall_upper <= 0.01,
        "termination_difference_ci_upper_le_0p01": termination_upper <= 0.01,
        "command_distribution_tv_le_0p02": tv <= 0.02,
        "no_resumed_only_failure": not (
            crash_r > crash_u
            or any(float(x["nonfinite_count"]) > 0 for x in r_rows)
        ),
    }
    decision = (
        "STATISTICAL_RESUME_PASS"
        if all(pass_checks.values())
        else "STATISTICAL_RESUME_FAIL"
    )
    summary = {
        "decision": decision,
        "completed_u": len(u_rows),
        "completed_r": len(r_rows),
        "crash_u": crash_u,
        "crash_r": crash_r,
        "nonfinite_count": nonfinite,
        "actor_delta_median_ratio_r_over_u": ratios["actor_delta_l2"],
        "critic_delta_median_ratio_r_over_u": ratios["critic_delta_l2"],
        "fall_rate_difference_ci95_upper": fall_upper,
        "termination_rate_difference_ci95_upper": termination_upper,
        "command_joint_distribution_total_variation": tv,
        "checks": pass_checks,
    }
    (args.output / "resume_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
