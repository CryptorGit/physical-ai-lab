"""Stratify the saved formal STOP evaluation without rerunning simulation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
csv.field_size_limit(16 * 1024 * 1024)
parser.add_argument("--evaluation", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)


def truth(value: str) -> bool:
    return value.lower() == "true"


def speed_stratum(speed: float) -> str:
    if speed <= 1.4:
        return "in_range_le_1.4"
    if speed <= 1.8:
        return "moderate_tail_1.4_1.8"
    return "high_tail_gt_1.8"


def aggregate(rows: list[dict], field: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return {
        name: {
            "episodes": len(group),
            "falls": sum(row["fall"] for row in group),
            "heading_failures": sum(row["failure_class"] == "heading_failure" for row in group),
            "success_rate": sum(row["success"] for row in group) / len(group),
            "fall_rate": sum(row["fall"] for row in group) / len(group),
            "ankle_saturation_rate": sum(row["ankle_saturation"] for row in group) / len(group),
            "yaw_max_rps": max(row["yaw_max_rps"] for row in group),
        }
        for name, group in sorted(groups.items())
    }


def main() -> None:
    args = parser.parse_args()
    evaluation = args.evaluation.resolve(strict=True)
    with (evaluation / "skills.csv").open(newline="", encoding="utf-8-sig") as stream:
        skills = list(csv.DictReader(stream))
    with (evaluation / "stop_curve.csv").open(newline="", encoding="utf-8-sig") as stream:
        curves = list(csv.DictReader(stream))
    by_episode_curve = defaultdict(list)
    for row in curves:
        by_episode_curve[int(row["episode"])].append(row)
    run_by_episode = {int(row["episode"]): row for row in skills if row["skill"] == "RUN"}
    stop_rows = [row for row in skills if row["skill"] == "STOP"]
    episodes = []
    for stop in stop_rows:
        episode = int(stop["episode"])
        curve = by_episode_curve[episode]
        supports = [int(float(point.get("feedback_support_count", 0))) for point in curve]
        phases = [point.get("stop_phase", "") for point in curve]
        braking_supports = [support for support, phase in zip(supports, phases) if phase == "braking"]
        transition = "none"
        transition_time = ""
        for index in range(1, len(supports)):
            if supports[index] != supports[index - 1]:
                transition = f"{supports[index - 1]}->{supports[index]}"
                transition_time = curve[index].get("time_s", "")
                break
        initial_support = supports[0] if supports else -1
        entry = float(stop["stop_entry_speed_mps"])
        run = run_by_episode.get(episode, {})
        episodes.append({
            "episode": episode,
            "failure_class": stop["failure_class"] or "success",
            "success": truth(stop["success"]),
            "fall": truth(stop["fall"]),
            "entry_speed_mps": entry,
            "entry_speed_stratum": speed_stratum(entry),
            "preceding_run_commanded_speed_mps": float(run.get("legacy_command_vx_mean", 0.0)),
            "stopping_distance_m": float(stop["stop_initial_distance_m"]),
            "initial_support_count": initial_support,
            "initial_support_class": {2: "double", 1: "single", 0: "flight"}.get(initial_support, "missing"),
            "initial_support_side_available": False,
            "first_support_transition": transition,
            "first_support_transition_time_s": transition_time,
            "braking_double_support_fraction": braking_supports.count(2) / max(len(braking_supports), 1),
            "braking_single_support_fraction": braking_supports.count(1) / max(len(braking_supports), 1),
            "braking_flight_fraction": braking_supports.count(0) / max(len(braking_supports), 1),
            "ankle_saturation": float(stop["ankle_torque_saturation_fraction"]) > 0.20,
            "joint_velocity_saturation": float(stop["joint_velocity_saturation_fraction"]) > 0.05,
            "spike_guard_fired": int(float(stop["feedback_spike_guard_count"])) > 0,
            "hard_guard_fired": int(float(stop["feedback_hard_guard_count"])) > 0,
            "yaw_max_rps": float(stop["actual_yaw_rate_abs_max_rps"]),
            "heading_mean_rad": float(stop["heading_error_rad"]),
        })
    failure_ids = [row["episode"] for row in episodes if row["failure_class"] != "success"]
    report = {
        "source": str(evaluation),
        "episode_count": len(episodes),
        "failure_episode_ids": failure_ids,
        "entry_speed": aggregate(episodes, "entry_speed_stratum"),
        "initial_support": aggregate(episodes, "initial_support_class"),
        "first_support_transition": aggregate(episodes, "first_support_transition"),
        "ankle_saturation": aggregate(episodes, "ankle_saturation"),
        "spike_guard": aggregate(episodes, "spike_guard_fired"),
        "hard_guard": aggregate(episodes, "hard_guard_fired"),
        "support_side_limitation": "formal_50 logged support count but not left/right identity",
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.resolve().with_suffix(".episodes.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
