#!/usr/bin/env python3
"""Offline-only v59 command_progress and tracking counterfactual surfaces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


YAW_RATIOS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
LINEAR_RATIOS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    manifest = json.loads(
        (root / "command_manifest.json").read_text(encoding="utf-8")
    )
    commands = {row["command_id"]: row for row in manifest["commands"]}
    reward_rows = list(
        csv.DictReader(
            (root / "reward_term_summary.csv").open(encoding="utf-8")
        )
    )
    empirical = {}
    for row in reward_rows:
        if row["condition"] == "D":
            empirical[(row["command_id"], row["term"])] = float(
                row["mean_per_step_pre_dt"]
            )

    output = []
    for command_id in ("C05_yaw_left", "C06_yaw_right"):
        command = commands[command_id]
        cmd_yaw = float(command["yaw_rate"])
        excluded = {
            "tracking_ang_vel",
            "command_yaw_error",
            "command_progress",
        }
        constant = sum(
            value
            for (cid, term), value in empirical.items()
            if cid == command_id and term not in excluded
        )
        for ratio in YAW_RATIOS:
            actual = ratio * cmd_yaw
            error = actual - cmd_yaw
            tracking_ang = 10.0 * np.exp(-(error**2) / 0.04)
            command_yaw_error = -20.0 * error**2
            command_progress = 100.0 * actual * cmd_yaw
            yaw_related = tracking_ang + command_yaw_error + command_progress
            total = constant + yaw_related
            output.append(
                {
                    "surface": "yaw_only",
                    "command_id": command_id,
                    "linear_tracking_ratio": 0.0,
                    "yaw_tracking_ratio": ratio,
                    "actual_vx": 0.0,
                    "actual_vy": 0.0,
                    "actual_yaw_rate": actual,
                    "tracking_lin_vel": empirical.get(
                        (command_id, "tracking_lin_vel"), 0.0
                    ),
                    "tracking_ang_vel": tracking_ang,
                    "command_velocity_error": empirical.get(
                        (command_id, "command_velocity_error"), 0.0
                    ),
                    "command_yaw_error": command_yaw_error,
                    "command_progress_linear": 0.0,
                    "command_progress_yaw": command_progress,
                    "command_progress_total": command_progress,
                    "constant_other_terms": constant,
                    "yaw_related_total": yaw_related,
                    "total_reward_pre_dt": total,
                    "total_reward_after_dt": total * 0.02,
                }
            )

    for command_id in (
        "C09_forward_yaw_left",
        "C11_forward_left_yaw_left",
        "C18_backward_yaw_right_0p3",
    ):
        command = commands[command_id]
        cmd_linear = np.asarray([command["vx"], command["vy"]], float)
        cmd_yaw = float(command["yaw_rate"])
        excluded = {
            "tracking_lin_vel",
            "tracking_ang_vel",
            "command_velocity_error",
            "command_yaw_error",
            "command_progress",
        }
        constant = sum(
            value
            for (cid, term), value in empirical.items()
            if cid == command_id and term not in excluded
        )
        sigma = 0.005 if command["vx"] < -0.02 else 0.02
        multiplier = 4.0 if command["vx"] < -0.02 else 2.0
        for linear_ratio in LINEAR_RATIOS:
            actual_linear = linear_ratio * cmd_linear
            linear_error = actual_linear - cmd_linear
            tracking_lin = (
                10.0
                * multiplier
                * np.exp(-np.sum(linear_error**2) / sigma)
            )
            command_velocity_error = -50.0 * np.sum(linear_error**2)
            progress_linear = 100.0 * float(
                np.dot(actual_linear, cmd_linear)
            )
            for yaw_ratio in YAW_RATIOS:
                actual_yaw = yaw_ratio * cmd_yaw
                yaw_error = actual_yaw - cmd_yaw
                tracking_ang = 10.0 * np.exp(-(yaw_error**2) / 0.04)
                command_yaw_error = -20.0 * yaw_error**2
                progress_yaw = 100.0 * actual_yaw * cmd_yaw
                progress = progress_linear + progress_yaw
                total = (
                    constant
                    + tracking_lin
                    + tracking_ang
                    + command_velocity_error
                    + command_yaw_error
                    + progress
                )
                output.append(
                    {
                        "surface": "compound",
                        "command_id": command_id,
                        "linear_tracking_ratio": linear_ratio,
                        "yaw_tracking_ratio": yaw_ratio,
                        "actual_vx": actual_linear[0],
                        "actual_vy": actual_linear[1],
                        "actual_yaw_rate": actual_yaw,
                        "tracking_lin_vel": tracking_lin,
                        "tracking_ang_vel": tracking_ang,
                        "command_velocity_error": command_velocity_error,
                        "command_yaw_error": command_yaw_error,
                        "command_progress_linear": progress_linear,
                        "command_progress_yaw": progress_yaw,
                        "command_progress_total": progress,
                        "constant_other_terms": constant,
                        "yaw_related_total": (
                            tracking_ang + command_yaw_error + progress_yaw
                        ),
                        "total_reward_pre_dt": total,
                        "total_reward_after_dt": total * 0.02,
                    }
                )
    write_csv(root / "command_progress_counterfactual.csv", output)

    conclusions = {}
    for command_id in (
        "C05_yaw_left",
        "C06_yaw_right",
        "C09_forward_yaw_left",
        "C11_forward_left_yaw_left",
        "C18_backward_yaw_right_0p3",
    ):
        rows = [row for row in output if row["command_id"] == command_id]
        conclusions[command_id] = {
            "command_progress_max": max(
                rows, key=lambda row: float(row["command_progress_total"])
            ),
            "yaw_related_total_max": max(
                rows, key=lambda row: float(row["yaw_related_total"])
            ),
            "total_reward_max": max(
                rows, key=lambda row: float(row["total_reward_pre_dt"])
            ),
        }
    (root / "command_progress_counterfactual_summary.json").write_text(
        json.dumps(conclusions, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
