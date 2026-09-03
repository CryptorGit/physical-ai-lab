"""Summarize CROUCH support, RETURN wiring, and joint saturation from curve CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


PHASES = {"0": "settle", "1": "down", "2": "hold", "3": "return", "4": "stand_hold"}
JOINTS = (
    "left_hip_pitch", "right_hip_pitch", "left_knee", "right_knee",
    "left_ankle_pitch", "right_ankle_pitch",
)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def numeric(rows, key):
    return [float(row[key]) for row in rows if row.get(key, "") != ""]


def summarize(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_phase = defaultdict(list)
    by_episode = defaultdict(list)
    for row in rows:
        by_phase[PHASES[row["phase"]]].append(row)
        by_episode[row["episode"]].append(row)

    support = {}
    wiring = {}
    saturation = {}
    for phase, selected in by_phase.items():
        support[phase] = {
            "steps": len(selected),
            "left_contact_fraction": mean([row["left_foot_contact"] == "True" for row in selected]),
            "right_contact_fraction": mean([row["right_foot_contact"] == "True" for row in selected]),
            "double_fraction": mean([row["support_state"] == "double" for row in selected]),
            "single_fraction": mean(["single" in row["support_state"] for row in selected]),
            "flight_fraction": mean([row["support_state"] == "flight" for row in selected]),
        }
        target = numeric(selected, "target_absolute_pelvis_height_m")
        relative = numeric(selected, "target_relative_height_m")
        actual = numeric(selected, "pelvis_height_m")
        progress = numeric(selected, "return_progress")
        gate = numeric(selected, "crouch_gate")
        wiring[phase] = {
            "target_absolute_start_m": target[0], "target_absolute_end_m": target[-1],
            "target_relative_start_m": relative[0], "target_relative_end_m": relative[-1],
            "actual_height_start_m": actual[0], "actual_height_end_m": actual[-1],
            "return_progress_min": min(progress), "return_progress_max": max(progress),
            "crouch_gate_min": min(gate), "crouch_gate_max": max(gate),
            "height_error_abs_mean_m": mean([abs(value) for value in numeric(selected, "height_error_m")]),
        }
        saturation[phase] = {}
        for joint in JOINTS:
            saturation[phase][joint] = {
                "residual_limit_fraction": mean([row[f"{joint}_residual_saturated"] == "True" for row in selected]),
                "velocity_utilization_ge_0_95_fraction": mean([
                    float(row[f"{joint}_velocity_utilization"]) >= 0.95 for row in selected
                ]),
                "torque_utilization_ge_0_95_fraction": mean([
                    float(row[f"{joint}_torque_utilization"]) >= 0.95 for row in selected
                ]),
                "residual_abs_mean": mean([abs(value) for value in numeric(selected, f"{joint}_residual")]),
            }

    commanded = numeric(rows, "commanded_vx_mps")
    speed = numeric(rows, "actual_forward_speed_mps")
    height = numeric(rows, "pelvis_height_m")
    switches = []
    for episode_rows in by_episode.values():
        states = [row["support_state"] for row in episode_rows]
        switches.append(sum(left != right for left, right in zip(states, states[1:])))
    return {
        "rows": len(rows),
        "episodes": len(by_episode),
        "base_behavior": {
            "commanded_vx_abs_mean_mps": mean([abs(value) for value in commanded]),
            "commanded_vx_abs_max_mps": max([abs(value) for value in commanded], default=0.0),
            "actual_forward_speed_abs_mean_mps": mean([abs(value) for value in speed]),
            "actual_forward_speed_abs_max_mps": max([abs(value) for value in speed], default=0.0),
            "pelvis_height_range_m": max(height) - min(height),
            "support_switches_mean_per_episode": mean(switches),
        },
        "phase_support": support,
        "return_wiring": wiring,
        "saturation_by_phase_and_joint": saturation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-0", type=Path, required=True)
    parser.add_argument("--model-31", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"model_0": summarize(args.model_0), "model_31": summarize(args.model_31)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
