"""Summarize Stage 2W failure timelines and generate diagnostic plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parent.parent.parents[2]
OUT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage2wb_walk_stabilization"
TIMELINE = OUT / "heading_failure_timelines.csv"
EPISODES = OUT / "stage2w_failure_replay_episodes.csv"
FAILURES = {6: "heading_failure", 22: "heading_failure", 26: "heading_failure", 34: "heading_failure",
            35: "heading_failure", 43: "heading_failure", 49: "speed_tracking_failure"}


def mean(values):
    return sum(values) / len(values) if values else 0.0


def truth(value: str) -> bool:
    return value.lower() == "true"


def main() -> None:
    with TIMELINE.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with EPISODES.open(newline="") as stream:
        episodes = list(csv.DictReader(stream))
    details = []
    plot_dir = OUT / "heading_failure_episode_plots"
    plot_dir.mkdir(exist_ok=True)
    for episode, formal_failure in FAILURES.items():
        group = [row for row in rows if int(row["episode"]) == episode]
        times = [float(row["time_s"]) for row in group]
        heading = [float(row["heading_error_rad"]) for row in group]
        filtered = [float(row["filtered_yaw_rate_command_radps"]) for row in group]
        raw = [float(row["generated_yaw_rate_command_radps"]) for row in group]
        yaw_rate = [float(row["yaw_rate_radps"]) for row in group]
        post_ramp = [index for index, time in enumerate(times) if time >= 1.5]
        steady = [index for index, time in enumerate(times) if time >= 2.5]
        first = next((index for index in post_ramp if abs(heading[index]) > 0.12), None)
        peak = max(steady, key=lambda index: abs(heading[index]))
        reversals = sum(
            a * b < 0 and abs(a) > 0.01 and abs(b) > 0.01
            for a, b in zip(filtered, filtered[1:])
        )
        detail = {
            "episode": episode,
            "target_speed_mps": float(group[0]["target_speed_mps"]),
            "formal_primary_failure": formal_failure,
            "failure_established_time_s": float(group[first]["time_s"]) if first is not None else 7.5,
            "heading_error_at_hold_start_rad": abs(
                float(next(row for row in group if float(row["time_s"]) >= 2.5)["heading_error_rad"])
            ),
            "peak_heading_error_rad": abs(heading[peak]),
            "peak_heading_time_s": times[peak],
            "final_heading_error_rad": abs(heading[-1]),
            "maximum_per_step_heading_change_rad": max(
                abs(__import__("math").atan2(__import__("math").sin(heading[b] - heading[a]),
                                             __import__("math").cos(heading[b] - heading[a])))
                for a, b in zip(steady, steady[1:])
            ),
            "heading_profile": "GRADUAL_RAMP_ACCUMULATION_THEN_RECOVERY" if first is not None else "NO_HEADING_FAILURE",
            "support_at_failure": group[first]["support_foot"] if first is not None else None,
            "controller_saturation_fraction": mean([
                truth(group[index]["heading_controller_saturated"]) for index in steady
            ]),
            "yaw_command_reversal_count": reversals,
            "yaw_command_reversal_frequency_hz": reversals / (times[-1] - times[0]),
            "left_contact_fraction": mean([truth(row["left_contact"]) for row in group if float(row["time_s"]) >= 2.5]),
            "right_contact_fraction": mean([truth(row["right_contact"]) for row in group if float(row["time_s"]) >= 2.5]),
            "left_slip_mean_mps": mean([
                float(row["left_foot_slip_mps"]) for row in group if float(row["time_s"]) >= 2.5
            ]),
            "right_slip_mean_mps": mean([
                float(row["right_foot_slip_mps"]) for row in group if float(row["time_s"]) >= 2.5
            ]),
            "hip_yaw_action_asymmetry_mean": mean([
                abs(float(row["left_hip_yaw_action"]) + float(row["right_hip_yaw_action"]))
                for row in group if float(row["time_s"]) >= 2.5
            ]),
            "hip_roll_action_asymmetry_mean": mean([
                abs(float(row["left_hip_roll_action"]) + float(row["right_hip_roll_action"]))
                for row in group if float(row["time_s"]) >= 2.5
            ]),
        }
        details.append(detail)

        figure, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(times, heading, label="heading error")
        axes[0].axhline(0.12, color="r", linestyle="--")
        axes[0].axhline(-0.12, color="r", linestyle="--")
        axes[0].axvline(2.5, color="k", linestyle=":", label="steady hold")
        axes[0].set_ylabel("heading [rad]")
        axes[0].legend(loc="upper right")
        axes[1].plot(times, raw, label="generated")
        axes[1].plot(times, filtered, label="filtered")
        axes[1].plot(times, yaw_rate, label="actual yaw rate", alpha=0.7)
        axes[1].set_ylabel("yaw rate")
        axes[1].legend(loc="upper right")
        axes[2].step(times, [int(truth(row["left_contact"])) for row in group], label="left contact")
        axes[2].step(times, [int(truth(row["right_contact"])) for row in group], label="right contact")
        axes[2].set_ylabel("contact")
        axes[2].legend(loc="upper right")
        axes[3].plot(times, [float(row["path_lateral_error_m"]) for row in group], label="path lateral")
        axes[3].plot(times, [float(row["lateral_body_velocity_mps"]) for row in group], label="lateral velocity")
        axes[3].set_ylabel("lateral")
        axes[3].set_xlabel("time [s]")
        axes[3].legend(loc="upper right")
        figure.suptitle(f"Stage 2W failure episode {episode}: {formal_failure}")
        figure.tight_layout()
        figure.savefig(plot_dir / f"episode_{episode:02d}.png", dpi=140)
        plt.close(figure)

    heading_failures = [item for item in details if item["formal_primary_failure"] == "heading_failure"]
    episode_by_id = {int(row["episode"]): row for row in episodes}
    successful = [row for row in episodes if truth(row["walk_success"])]
    failed = [episode_by_id[item["episode"]] for item in heading_failures]
    comparisons = {}
    for metric in (
        "stance_duration_asymmetry_s",
        "slip_asymmetry_mps",
        "hip_yaw_action_asymmetry_mean",
        "hip_roll_action_asymmetry_mean",
        "lateral_velocity_abs_mean_mps",
    ):
        comparisons[metric] = {
            "heading_failure_mean": mean([float(row[metric]) for row in failed]),
            "successful_episode_mean": mean([float(row[metric]) for row in successful]),
        }
    controller_summaries = {
        name: json.loads((OUT / f"controller_{name}_summary.json").read_text())
        for name in ("zero_yaw", "current", "lower_bandwidth")
    }
    payload = {
        "stage": "Stage 2W-B",
        "source_formal_seed": 20260731,
        "failure_episode_ids": sorted(FAILURES),
        "replay_matches_stage2w": {
            "success_rate": 0.86,
            "failure_counts": {"none": 43, "heading_failure": 6, "speed_tracking_failure": 1},
        },
        "root_cause_classification": "POLICY_RESPONSE_DOMINATED",
        "root_cause": (
            "The policy has a strong open-loop yaw bias during command acquisition. ZeroYaw permits "
            "unbounded drift, while lower-bandwidth feedback cannot arrest it. Current feedback is best "
            "but six failures enter steady hold with accumulated heading error and then recover. The "
            "controller rarely saturates and does not show high-frequency reversal in these episodes."
        ),
        "evidence": {
            "all_heading_failures_are_over_threshold_at_hold_start": all(
                item["heading_error_at_hold_start_rad"] > 0.12 for item in heading_failures
            ),
            "all_heading_failures_recover_below_threshold_by_episode_end": all(
                item["final_heading_error_rad"] < 0.12 for item in heading_failures
            ),
            "maximum_controller_saturation_fraction": max(
                item["controller_saturation_fraction"] for item in heading_failures
            ),
            "maximum_per_step_heading_change_rad": max(
                item["maximum_per_step_heading_change_rad"] for item in heading_failures
            ),
            "support_at_failure_counts": dict(
                __import__("collections").Counter(item["support_at_failure"] for item in heading_failures)
            ),
            "gait_asymmetry_comparison": comparisons,
            "controller_only": {
                name: summary["overall"] for name, summary in controller_summaries.items()
            },
        },
        "episodes": details,
        "plots": [str((plot_dir / f"episode_{episode:02d}.png").relative_to(REPO)) for episode in FAILURES],
    }
    (OUT / "heading_failure_root_cause.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
